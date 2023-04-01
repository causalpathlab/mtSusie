#include "mtSusie.hh"

//' Estimate mtSusie regression model with an interaction matrix
//'
//' @param x          design matrix
//' @param y          output matrix
//' @param w          interaction matrix
//' @param levels     number of "single" effects
//' @param max_iter   maximum iterations
//' @param min_iter   minimum iterations
//' @param tol        tolerance
//' @param prior_var  prior variance
//' @param lodds_cutoff log-odds cutoff in trait selection steps
//' @param min_pip_cutoff minimum PIP cutoff in building credible sets
//' @param full_stat keep full statistics
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
//' \item{variants}{variant indexes `[1 .. p]`}
//' \item{traits}{trait indexes `[1 .. m]`}
//' \item{levels}{levels `[1 .. L]`}
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
                         const std::size_t levels_per_inter = 3,
                         const std::size_t max_iter = 100,
                         const std::size_t min_iter = 1,
                         const double coverage = .9,
                         const double tol = 1e-4,
                         const double prior_var = 0.1,
                         const double lodds_cutoff = 0,
                         Rcpp::Nullable<double> min_pip_cutoff = R_NilValue,
                         const bool full_stat = true)
{

    const Mat xx = Rcpp::as<Mat>(x), yy = Rcpp::as<Mat>(y);
    const Mat ww = Rcpp::as<Mat>(w);

    const std::size_t levels = levels_per_inter * (ww.cols() + 1);

    shared_regression_t model(yy.rows(),
                              yy.cols(),
                              xx.cols(),
                              levels,
                              prior_var);

    shared_effect_stat_t stat(xx.cols(), yy.cols());

    ASSERT_RETL(ww.rows() == xx.rows(),
                "w and x have different numbers of rows");
    ASSERT_RETL(yy.rows() == xx.rows(),
                "y and x have different numbers of rows");

    std::vector<Scalar> score;
    score.reserve(max_iter);

    /////////////////////////////////////////////////////
    // Mark which level will involve interaction terms //
    /////////////////////////////////////////////////////

    const Index L = model.lvl;
    const Index K = ww.cols();
    std::vector<int> interaction(L);
    for (Index l = 0; l < L; ++l) {
        if (l < (levels_per_inter * K)) {
            const Index k = std::floor(l / levels_per_inter);
            interaction[l] = k; // it's an interaction term with k-th var.
        } else {
            interaction[l] = -1; // not an interaction term
        }

        TLOG("level [" << l << "] using "
                       << " interaction term " << interaction[l]);
    }

    TLOG(K << " interaction variables, total " << L << " levels");

    for (Index iter = 0; iter < max_iter; ++iter) {
        const Scalar curr =
            update_shared_interaction_regression(model,
                                                 stat,
                                                 xx,
                                                 yy,
                                                 ww,
                                                 interaction,
                                                 levels_per_inter,
                                                 lodds_cutoff);

        if (iter > min_iter) {
            const Scalar prev = score.at(score.size() - 1);
            const Scalar diff = std::abs(prev - curr) / (std::abs(curr) + 1e-8);
            if (diff < tol) {
                score.emplace_back(curr);
                TLOG("Converged at " << iter << ", " << curr);
                break;
            }
        }
        TLOG("mtSusie [" << iter << "] " << curr);
        score.emplace_back(curr);
    }

    TLOG("Exporting model estimation results");

    using ivec = std::vector<int>;
    Rcpp::List ret = mt_susie_output(model, full_stat);

    {
        ivec inter_out;
        inter_out.reserve(interaction.size()); // give interaction term IDs
        for (auto i : interaction)             //
            inter_out.emplace_back(i + 1);     // 1-based
        ret["interaction"] = inter_out;        //
    }

    TLOG("Sorting credible sets");

    const Scalar _pip_cutoff = min_pip_cutoff.isNotNull() ?
        Rcpp::as<Scalar>(min_pip_cutoff) :
        (1. / static_cast<Scalar>(xx.cols()));

    Rcpp::List cs = mt_susie_credible_sets(model, coverage, _pip_cutoff);
    {
        auto level_vec = Rcpp::as<ivec>(cs["levels"]);
        ivec inter_out;
        inter_out.reserve(level_vec.size());            //
        for (auto l : level_vec) {                      //
            inter_out.emplace_back(interaction[l] + 1); // 1-based
        }
        cs["interaction"] = inter_out;
    }
    ret["cs"] = cs;

    TLOG("Done");
    return ret;
}
