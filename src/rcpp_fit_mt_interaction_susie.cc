#include "mtSusie.hh"

//' Estimate mtSusie regression model with an interaction matrix
//'
//' @param x                design matrix
//' @param y                output matrix
//' @param w                interaction matrix
//' @param levels_per_wx    number of "single" effects for w*x terms
//' @param levels_per_w     number of "single" effects for w terms
//' @param levels_per_x     number of "single" effects for x terms
//' @param max_iter         maximum iterations
//' @param min_iter         minimum iterations
//' @param tol              tolerance
//' @param prior_var        prior variance
//' @param min_pip_cutoff   minimum PIP cutoff in building credible sets
//' @param full_stat        keep full statistics
//' @param update_prior     update prior variance or not
//'
//'
//' @return a list of mtSusie results
//' \item{alpha}{Posterior probability of variant across `p x level`}
//' \item{resid.var}{residual variance `level x traits`}
//' \item{prior.var}{prior variance `1 x traits`}
//' \item{log.odds}{log-odds ratio `level x traits`}
//' \item{mu}{a list of `p x m` mean parameters}
//' \item{var}{a list of `p x m` variance parameters}
//' \item{lbf}{a list of `p x m` log-Bayes Factors}
//' \item{z}{a list of `p x m` z-scores}
//' \item{n}{number of samples}
//' \item{m}{number of traits/outputs}
//' \item{p}{number of variants}
//' \item{L}{number of layers/levels}
//'
//' \item{cs}{credible sets (use `data.table::setDT` to assemble)}
//'
//' In the `cs`, we have
//' \item{variant}{variant indexes `[1 .. p]`}
//' \item{trait}{trait indexes `[1 .. m]`}
//' \item{level}{levels `[1 .. L]`}
//' \item{alpha}{shared PIP}
//' \item{mean}{univariate mean}
//' \item{var}{univariate var}
//' \item{z}{univariate z-score}
//' \item{lfsr}{local false sign rate}
//' \item{lodds}{log-odds ratio}
//' \item{interaction}{interacting column index}
//'
// [[Rcpp::export]]
Rcpp::List
fit_mt_interaction_susie(const Rcpp::NumericMatrix x,
                         const Rcpp::NumericMatrix y,
                         const Rcpp::NumericMatrix w,
                         const std::size_t levels_per_wx = 3,
                         const std::size_t levels_per_w = 1,
                         const std::size_t levels_per_x = 1,
                         const std::size_t max_iter = 100,
                         const std::size_t min_iter = 1,
                         const double coverage = .9,
                         const double tol = 1e-4,
                         const double prior_var = 0.1,
                         Rcpp::Nullable<double> min_pip_cutoff = R_NilValue,
                         const bool full_stat = true,
                         const bool update_prior = false,
                         const bool do_hard_selection = false,
                         const double hard_lodds_cutoff = 0.)
{

    const Mat xx = Rcpp::as<Mat>(x), yy = Rcpp::as<Mat>(y);
    const Mat ww = Rcpp::as<Mat>(w);

    ASSERT_RETL(ww.rows() == xx.rows(),
                "w and x have different numbers of rows");
    ASSERT_RETL(yy.rows() == xx.rows(),
                "y and x have different numbers of rows");

    ASSERT_RETL(levels_per_wx > 0, "at least one per W*X term");
    ASSERT_RETL(levels_per_x >= 0, "non-negative per X term");
    ASSERT_RETL(levels_per_w >= 0, "non-negative one per W term");

    Mat Y = yy;                   // a working response matrix
    Mat WX(xx.rows(), xx.cols()); // temporary w[i,k] * x[i,j]
    WX.setZero();

    shared_regression_t model_w(Y.rows(),     //
                                Y.cols(),     // Y ~ W
                                ww.cols(),    //
                                levels_per_w, //
                                prior_var);

    shared_effect_stat_t stat_w(ww.cols(), yy.cols());
    set_prior_var(stat_w, prior_var);

    shared_regression_t model_x(Y.rows(),     //
                                Y.cols(),     // Y ~ X
                                xx.cols(),    //
                                levels_per_x, //
                                prior_var);

    shared_effect_stat_t stat_x(xx.cols(), yy.cols());
    set_prior_var(stat_x, prior_var);

    const std::size_t wx_levels = levels_per_wx * ww.cols();

    shared_regression_t model_wx(Y.rows(),  //
                                 Y.cols(),  // Y ~ X
                                 xx.cols(), //
                                 wx_levels, //
                                 prior_var);

    shared_effect_stat_t stat_wx(xx.cols(), yy.cols());
    set_prior_var(stat_wx, prior_var);

    std::vector<Scalar> score_vec;
    score_vec.reserve(max_iter);

    for (Index iter = 0; iter < max_iter; ++iter) {

        //////////////////////////
        // 1. Regress out Y ~ W //
        //////////////////////////
        Scalar score = 0.;
        {
            Y = yy - model_x.fitted_nm - model_wx.fitted_nm;

            for (Index l = 0; l < levels_per_w; ++l) { //
                discount_model_stat(model_w,           // a. discount model stat
                                    ww,                //
                                    Y,                 //
                                    l);                //
                                                       //
                score += SER(ww,                 // b. update sufficient stat
                             model_w.partial_nm, //   - partial prediction
                             model_w.residvar_m, //   - residual variance
                             model_w.get_v0(l),  //   - prior variance
                             stat_w,             //   - statistics
                             update_prior);      //
                                                 //
                update_model_stat(model_w, // c. Put back the updated stats
                                  ww,      // design matrix
                                  stat_w,  // new stat
                                  l);      // level l
            }

            calibrate_residual_variance(model_w, Y);
        } // end of Y ~ W

        //////////////////////////
        // 2. Regress out Y ~ X //
        //////////////////////////
        {
            Y = yy - model_w.fitted_nm - model_wx.fitted_nm;

            for (Index l = 0; l < levels_per_x; ++l) { //
                discount_model_stat(model_x,           // a. discount model stat
                                    xx,                //
                                    Y,                 //
                                    l);                //
                                                       //
                score += SER(xx,                 // b. update sufficient stat
                             model_x.partial_nm, //   - partial prediction
                             model_x.residvar_m, //   - residual variance
                             model_x.get_v0(l),  //   - prior variance
                             stat_x,             //   - statistics
                             update_prior);      //
                                                 //
                update_model_stat(model_x, // c. Put back the updated stats
                                  xx,      // design matrix
                                  stat_x,  // new stat
                                  l);      // level l
            }
            calibrate_residual_variance(model_x, Y);
        } // end of Y ~ X

        //////////////////////////
        // 3. Regress Y ~ X * W //
        //////////////////////////
        {
            Y = yy - model_w.fitted_nm - model_x.fitted_nm;
            score = 0.;

            for (Index l = 0; l < levels_per_wx; ++l) {
                for (Index k = 0; k < ww.cols(); ++k) {
                    WX.setZero(); //
                    WX = xx.array().colwise() *
                        ww.col(k).array(); // weighted by W[,k]
                }
                discount_model_stat(model_wx,     // a. discount model stat
                                    WX,           //
                                    Y,            //
                                    l);           //
                                                  //
                score += SER(WX,                  // b. update sufficient stat
                             model_wx.partial_nm, //   - partial prediction
                             model_wx.residvar_m, //   - residual variance
                             model_wx.get_v0(l),  //   - prior variance
                             stat_wx,             //   - statistics
                             update_prior);       //
                                                  //
                update_model_stat(model_wx,       // c. Put back the updated
                                  WX,             // design matrix
                                  stat_wx,        // new stat
                                  l);             // level l
            }
            calibrate_residual_variance(model_wx, Y);
        } // end of Y ~ W * X

        if (iter > min_iter) {
            Scalar prev = score_vec.at(score_vec.size() - 1);
            Scalar diff = std::abs(prev - score) / (std::abs(score) + 1e-8);
            if (diff < tol) {
                score_vec.emplace_back(score);
                TLOG("Converged at " << iter << ", " << score);
                break;
            }
        }

        TLOG("mtSusie [" << iter << "] " << score);
        score_vec.emplace_back(score);

        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            WLOG("Interruption by the user at t=" << iter);
            break;
        }
    }

    TLOG("Exporting model estimation results");

    Rcpp::List ret = mt_susie_output(model_wx, full_stat);

    TLOG("Sorting credible sets");

    const Scalar _pip_cutoff = min_pip_cutoff.isNotNull() ?
        Rcpp::as<Scalar>(min_pip_cutoff) :
        (1. / static_cast<Scalar>(xx.cols()));

    std::vector<int> inter_names;
    for (Index l = 0; l < levels_per_wx; ++l) {
        for (Index k = 0; k < ww.cols(); ++k) {
            inter_names.emplace_back(k + 1);
        }
    }
    ret["interaction"] = inter_names;

    Rcpp::List cs = mt_susie_credible_sets(model_wx, coverage, _pip_cutoff);
    ret["cs"] = cs;
    ret["score"] = score_vec;
    ret["fitted"] = model_fitted(model_wx);
    ret["residuals"] = model_residuals(model_wx, yy);

    TLOG("Done");
    return ret;
}
