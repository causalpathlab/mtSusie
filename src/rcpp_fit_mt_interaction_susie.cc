#include "mtSusie.hh"

// [[Rcpp::export]]
Rcpp::List
fit_mt_interaction_susie(const Rcpp::NumericMatrix &x,
                         const Rcpp::NumericMatrix &y,
                         const Rcpp::NumericMatrix &w,
                         const std::size_t levels_per_inter = 3,
                         const std::size_t max_iter = 100,
                         const std::size_t min_iter = 5,
                         const double coverage = .9,
                         const double tol = 1e-8,
                         const double prior_var = 100.0,
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

    std::vector<Scalar> loglik;
    loglik.reserve(max_iter);

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
