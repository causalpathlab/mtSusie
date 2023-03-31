#include "mtSusie.hh"

//' Estimate mtSusie regression model with counterfactual data
//'
//' @param x1         factual design matrix
//' @param y1         factual output matrix
//' @param x0         counterfactual design matrix
//' @param y0         counterfactual output matrix
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
//'
// [[Rcpp::export]]
Rcpp::List
fit_mt_cf_susie(const Rcpp::NumericMatrix x1,
                const Rcpp::NumericMatrix y1,
                const Rcpp::NumericMatrix x0,
                const Rcpp::NumericMatrix y0,
                const std::size_t levels = 15,
                const std::size_t max_iter = 100,
                const std::size_t min_iter = 1,
                const double coverage = .9,
                const double tol = 1e-8,
                const double prior_var = 0.01,
                const double lodds_cutoff = 0,
                Rcpp::Nullable<double> min_pip_cutoff = R_NilValue,
                const bool full_stat = true)
{

    shared_regression_t model(y1.rows(),
                              y1.cols(),
                              x1.cols(),
                              levels,
                              prior_var);

    shared_effect_stat_t stat(x1.cols(), y1.cols());

    shared_regression_t cf_model(y0.rows(),
                                 y0.cols(),
                                 x0.cols(),
                                 levels,
                                 prior_var);

    shared_effect_stat_t cf_stat(x0.cols(), y0.cols());

    const Mat xx = Rcpp::as<Mat>(x1), yy = Rcpp::as<Mat>(y1);
    const Mat xx0 = Rcpp::as<Mat>(x0), yy0 = Rcpp::as<Mat>(y0);

    ASSERT_RETL(yy.rows() == xx.rows(),
                "y and x have different numbers of rows");

    std::vector<Scalar> score;
    score.reserve(max_iter);

    for (Index iter = 0; iter < max_iter; ++iter) {

        Scalar curr;
        curr = update_paired_regression(model,
                                        cf_model,
                                        stat,
                                        cf_stat,
                                        xx,
                                        yy,
                                        xx0,
                                        yy0,
                                        lodds_cutoff);

        if (iter > min_iter) {
            const Scalar prev = score.at(score.size() - 1);
            const Scalar diff = std::abs(curr - prev) / (std::abs(curr) + 1e-8);
            if (diff < tol) {
                score.emplace_back(curr);
                TLOG("Converged at " << iter << ", " << curr);
                break;
            }
        }
        TLOG("mtSusie [" << iter << "] " << curr);
        score.emplace_back(curr);
    }

    // Prune out the alpha (pip) scores of the factual model
    // using the counterfactual alpha scores
    for (Index l = 0; l < model.lvl; ++l) {
        for (Index k = 0; k < cf_model.lvl; ++k) {
            model.shared_pip_pl.col(l).array() *=
                (1.0 - cf_model.shared_pip_pl.col(k).array());
        }
    }

    TLOG("Exporting model estimation results");

    Rcpp::List ret = mt_susie_output(model, full_stat);
    ret["score"] = score;

    TLOG("Sorting credible sets");
    const Scalar _pip_cutoff = min_pip_cutoff.isNotNull() ?
        Rcpp::as<Scalar>(min_pip_cutoff) :
        (1. / static_cast<Scalar>(xx.cols()));

    ret["cs"] = mt_susie_credible_sets(model, coverage, _pip_cutoff);

    TLOG("Done");
    return ret;
}
