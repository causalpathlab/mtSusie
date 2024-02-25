#include "mtSusie.hh"

//' Estimate a multi-trait Sum of Single Effect regression model
//'
//' @param x          design matrix
//' @param y          output matrix
//' @param levels     number of "single" effects
//' @param max_iter   maximum iterations
//' @param min_iter   minimum iterations
//' @param tol        tolerance
//' @param prior_var  prior variance
//' @param lodds_cutoff log-odds cutoff in trait selection steps
//' @param min_pip_cutoff minimum PIP cutoff in building credible sets
//' @param full_stat keep full statistics
//' @param local_residual locally calculate residuals
//' @param update_prior update prior variance or not
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
//'
// [[Rcpp::export]]
Rcpp::List
fit_mt_susie(const Rcpp::NumericMatrix x,
             const Rcpp::NumericMatrix y,
             const std::size_t levels = 15,
             const std::size_t max_iter = 100,
             const std::size_t min_iter = 1,
             const double coverage = .9,
             const double tol = 1e-4,
             const double prior_var = 0.1,
             const double lodds_cutoff = 0,
             Rcpp::Nullable<double> min_pip_cutoff = R_NilValue,
             const bool full_stat = true,
             const bool local_residual = false,
             const bool update_prior = false)
{

    shared_regression_t model(y.rows(), y.cols(), x.cols(), levels, prior_var);
    shared_effect_stat_t stat(x.cols(), y.cols());
    set_prior_var(stat, prior_var);

    const Mat xx = Rcpp::as<Mat>(x), yy = Rcpp::as<Mat>(y);

    ASSERT_RETL(yy.rows() == xx.rows(),
                "y and x have different numbers of rows");

    std::vector<Scalar> score;
    score.reserve(max_iter + 1);

    ////////////////////////////////
    // Iterative Bayesian updates //
    ////////////////////////////////

    for (Index iter = 0; iter < max_iter; ++iter) {

        Scalar curr;
        curr = update_shared_regression(model,
                                        stat,
                                        xx,
                                        yy,
                                        local_residual,
                                        update_prior);

        if (iter > min_iter) {
            Scalar prev = score.at(score.size() - 1);
            Scalar diff = std::abs(prev - curr) / (std::abs(curr) + 1e-8);
            if (diff < tol) {
                score.emplace_back(curr);
                TLOG("Converged at " << iter << ", " << curr);
                break;
            }
        }
        TLOG("mtSusie [" << iter << "] " << curr);
        score.emplace_back(curr);

        try {
            Rcpp::checkUserInterrupt();
        } catch (Rcpp::internal::InterruptedException e) {
            WLOG("Interruption by the user at t=" << iter);
            break;
        }
    }

    TLOG("Exporting model estimation results");

    Rcpp::List ret = mt_susie_output(model, full_stat);

    TLOG("Sorting credible sets");
    const Scalar _pip_cutoff = min_pip_cutoff.isNotNull() ?
        Rcpp::as<Scalar>(min_pip_cutoff) :
        (1. / static_cast<Scalar>(xx.cols()));

    ret["cs"] = mt_susie_credible_sets(model, coverage, _pip_cutoff);
    ret["score"] = score;
    ret["fitted"] = model_fitted(model);
    ret["residuals"] = model_residuals(model, yy);

    TLOG("Done");
    return ret;
}
