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
//' \item{loglik}{log-likelihood trace}
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
//'
// [[Rcpp::export]]
Rcpp::List
fit_mt_susie(const Rcpp::NumericMatrix &x,
             const Rcpp::NumericMatrix &y,
             const std::size_t levels = 15,
             const std::size_t max_iter = 100,
             const std::size_t min_iter = 5,
             const double coverage = .9,
             const double tol = 1e-8,
             const double prior_var = 100.0,
             const double lodds_cutoff = 0,
             Rcpp::Nullable<double> min_pip_cutoff = R_NilValue,
             const bool full_stat = true)
{

    shared_regression_t model(y.rows(), y.cols(), x.cols(), levels, prior_var);
    shared_effect_stat_t stat(x.cols(), y.cols());

    const Mat xx = Rcpp::as<Mat>(x), yy = Rcpp::as<Mat>(y);

    ASSERT_RETL(yy.rows() == xx.rows(),
                "y and x have different numbers of rows");

    std::vector<Scalar> loglik;
    loglik.reserve(max_iter);

    for (Index iter = 0; iter < max_iter; ++iter) {

        const Scalar curr =
            update_shared_regression(model, stat, xx, yy, lodds_cutoff);

        if (iter >= min_iter) {
            const Scalar prev = loglik.at(loglik.size() - 1);
            const Scalar diff = std::abs(curr - prev) / (std::abs(curr) + 1e-8);
            if (diff < tol) {
                loglik.emplace_back(curr);
                TLOG("Converged at " << iter << ", " << curr);
                break;
            }
        }
        TLOG("mtSusie [" << iter << "] " << curr);
        loglik.emplace_back(curr);
    }

    TLOG("Exporting model estimation results");

    Rcpp::List ret = mt_susie_output(model, full_stat);
    ret["loglik"] = loglik;

    TLOG("Sorting credible sets");
    const Scalar _pip_cutoff = min_pip_cutoff.isNotNull() ?
        Rcpp::as<Scalar>(min_pip_cutoff) :
        (1. / static_cast<Scalar>(xx.cols()));

    ret["cs"] = mt_susie_credible_sets(model, coverage, _pip_cutoff);

    TLOG("Done");
    return ret;
}
