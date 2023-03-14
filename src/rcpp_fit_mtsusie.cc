#include "mtSusie.hh"

//' Simply export full statistics to Rcpp::List
//' @param model
template <typename MODEL>
Rcpp::List
mt_susie_output(const MODEL &model)
{
    return Rcpp::List::create(Rcpp::_["alpha"] = model.shared_pip_pl,
                              Rcpp::_["resid.var"] = model.resid_var_lm,
                              Rcpp::_["prior.var"] = model.prior_var,
                              Rcpp::_["mu"] = model.mu_pm_list,
                              Rcpp::_["var"] = model.var_pm_list,
                              Rcpp::_["lbf"] = model.lbf_pm_list,
                              Rcpp::_["z"] = model.z_pm_list,
                              Rcpp::_["n"] = model.n,
                              Rcpp::_["m"] = model.m,
                              Rcpp::_["p"] = model.p,
                              Rcpp::_["L"] = model.lvl);
}

//' Calibrate credible sets per level
//' @param model
//' @param coverage
//'
template <typename MODEL>
Rcpp::List
mt_susie_credible_sets(MODEL &model, const Scalar coverage)
{
    using svec = std::vector<Scalar>;
    using ivec = std::vector<Index>;
    svec mean_list, var_list, lbf_list, alpha_list, z_list, lfsr_list;
    ivec variants, traits, levels;

    const Index p = model.p, m = model.m;

    auto error_p = [](Scalar m, Scalar v) {
        Scalar pr = R::pnorm(0, m, std::sqrt(v), true, false);
        if (pr < .5)
            pr = 1. - pr;
        return pr;
    };

    for (Index l = 0; l < model.lvl; ++l) {
        std::vector<Scalar> alpha = std_vector(model.shared_pip_pl.col(l));
        std::vector<Index> order = std_argsort(alpha, true);

        Mat &mean = model.get_mean(l);
        Mat &var = model.get_var(l);
        Mat &lbf = model.get_lbf(l);
        Mat &zz = model.get_z(l);

        //////////////////////////////////////
        // Calibrate local false sign rates //
        //////////////////////////////////////

        // Mat error = mean.binaryExpr(var, error_p);
        Vec lfsr = -mean.binaryExpr(var, error_p).transpose() *
            model.shared_pip_pl.col(l);
        lfsr.array() += 1;

        //////////////////////////
        // Expand credible sets //
        //////////////////////////

        Scalar cum = 0;

        for (Index i = 0; i < model.p; ++i) {
            const Index j = order.at(i);
            cum += alpha.at(j);
            for (Index t = 0; t < model.m; ++t) {
                variants.emplace_back(j + 1); // 1-based
                traits.emplace_back(t + 1);   // 1-based
                levels.emplace_back(l + 1);   // 1-based
                alpha_list.emplace_back(alpha.at(j));
                mean_list.emplace_back(mean(j, t));
                var_list.emplace_back(var(j, t));
                lbf_list.emplace_back(lbf(j, t));
                z_list.emplace_back(zz(j, t));
                lfsr_list.emplace_back(lfsr(t));
            }
            if (cum > coverage)
                break;
        }
    }

    return Rcpp::List::create(Rcpp::_["variants"] = variants,
                              Rcpp::_["traits"] = traits,
                              Rcpp::_["levels"] = levels,
                              Rcpp::_["alpha"] = alpha_list,
                              Rcpp::_["mean"] = mean_list,
                              Rcpp::_["var"] = var_list,
                              Rcpp::_["lbf"] = lbf_list,
                              Rcpp::_["z"] = z_list,
                              Rcpp::_["lfsr"] = lfsr_list);
}

//' Estimate a multi-trait Sum of Single Effect regression model
//'
//' @param x          design matrix
//' @param y          output matrix
//' @param levels     number of "single" effects
//' @param max_iter   maximum iterations
//' @param min_iter   minimum iterations
//' @param tol        tolerance
//' @param prior_var  prior variance
//'
//' @return a list of mtSusie results
//'
//' \item{alpha}{Posterior probability of variant across k traits; `alpha[j,l]`}
//' \item{resid.var}{Residual variance; `alpha[l,k]`}
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
             const double lodds_cutoff = 0)
{

    shared_regression_t model(y.rows(), y.cols(), x.cols(), levels, prior_var);
    shared_effect_stat_t stat(x.cols(), y.cols());

    std::vector<Scalar> loglik;
    loglik.reserve(max_iter);

    const Mat xx = Rcpp::as<Mat>(x), yy = Rcpp::as<Mat>(y);

    for (Index iter = 0; iter < max_iter; ++iter) {

        const Scalar curr =
            update_shared_regression(model, stat, xx, yy, lodds_cutoff);

        if (iter >= min_iter) {
            const Scalar prev = loglik.at(loglik.size() - 1);
            const Scalar diff = std::abs(curr - prev) / std::abs(curr + 1e-8);
            if (diff < tol) {
                loglik.emplace_back(curr);
                TLOG("Converged at " << iter << ", " << curr);
                break;
            }
        }
        TLOG("mtSusie [" << iter << "] " << curr);
        loglik.emplace_back(curr);
    }

    Rcpp::List ret = mt_susie_output(model);
    ret["loglik"] = loglik;
    ret["cs"] = mt_susie_credible_sets(model, coverage);
    return ret;
}
