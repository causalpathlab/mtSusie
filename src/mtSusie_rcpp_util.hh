#ifndef MTSUSIE_RCPP_UTIL_HH_
#define MTSUSIE_RCPP_UTIL_HH_

//' Simply export full statistics to Rcpp::List
//' @param model susie regression model
//' @param full_stat full output or not
template <typename MODEL>
Rcpp::List
mt_susie_output(const MODEL &model, bool full_stat)
{
    if (full_stat) {
        return Rcpp::List::create(Rcpp::_["alpha.p"] = model.shared_pip_pl,
                                  Rcpp::_["alpha.m"] = model.shared_pip_lm,
                                  Rcpp::_["resid.var"] = model.residvar_m,
                                  Rcpp::_["prior.var"] = model.prior_var,
                                  Rcpp::_["log.odds"] = model.lodds_lm,
                                  Rcpp::_["mu"] = model.mu_pm_list,
                                  Rcpp::_["var"] = model.var_pm_list,
                                  Rcpp::_["lbf"] = model.lbf_pm_list,
                                  Rcpp::_["z"] = model.z_pm_list,
                                  Rcpp::_["n"] = model.n,
                                  Rcpp::_["m"] = model.m,
                                  Rcpp::_["p"] = model.p,
                                  Rcpp::_["L"] = model.lvl);
    } else {
        return Rcpp::List::create(Rcpp::_["alpha.p"] = model.shared_pip_pl,
                                  Rcpp::_["alpha.m"] = model.shared_pip_lm,
                                  Rcpp::_["resid.var"] = model.residvar_m,
                                  Rcpp::_["prior.var"] = model.prior_var,
                                  Rcpp::_["log.odds"] = model.lodds_lm,
                                  Rcpp::_["n"] = model.n,
                                  Rcpp::_["m"] = model.m,
                                  Rcpp::_["p"] = model.p,
                                  Rcpp::_["L"] = model.lvl);
    }
}

//' Calibrate credible sets per level
//' @param model susie regression model
//' @param coverage targeted coverage
//' @param pip_cutoff pip cutoff to reduce outputs
//'
template <typename MODEL>
Rcpp::List
mt_susie_credible_sets(MODEL &model,
                       const Scalar coverage,
                       const Scalar pip_cutoff,
                       const Scalar v0 = 1e-8)
{
    using svec = std::vector<Scalar>;
    using ivec = std::vector<Index>;
    svec mean_list, sd_list, lbf_list, alpha_list, z_list;
    svec lfsr_list, lodds_list;
    ivec variants, traits, levels;

    const Index p = model.p, m = model.m;

    //////////////////////////////////////
    // Calibrate local false sign rates //
    //////////////////////////////////////

    calibrate_lfsr(model, v0);

    for (Index l = 0; l < model.lvl; ++l) {
        std::vector<Scalar> alpha = std_vector(model.shared_pip_pl.col(l));
        std::vector<Index> order = std_argsort(alpha, true);

        Mat &mean = model.get_mean(l);
        Mat &var = model.get_var(l);
        Mat &lbf = model.get_lbf(l);
        Mat &zz = model.get_z(l);
        Mat &lodds = model.lodds_lm;
        Mat &lfsr = model.lfsr_lm;

        //////////////////////////
        // Expand credible sets //
        //////////////////////////

        Scalar cum = 0;

        for (Index i = 0; i < model.p; ++i) {
            const Index j = order.at(i);
            cum += alpha.at(j);
            if (alpha.at(j) < pip_cutoff)
                break;
            for (Index t = 0; t < model.m; ++t) {
                const Scalar a_t = model.shared_pip_lm(l, t);
                variants.emplace_back(j + 1); // 1-based
                traits.emplace_back(t + 1);   // 1-based
                levels.emplace_back(l + 1);   // 1-based
                alpha_list.emplace_back(alpha.at(j) * a_t);
                mean_list.emplace_back(mean(j, t));
                sd_list.emplace_back(std::sqrt(var(j, t)));
                lbf_list.emplace_back(lbf(j, t));
                z_list.emplace_back(zz(j, t));
                lfsr_list.emplace_back(lfsr(l, t));
                lodds_list.emplace_back(lodds(l, t));
            }
            if (cum > coverage)
                break;
        }
    }

    return Rcpp::List::create(Rcpp::_["variant"] = variants,
                              Rcpp::_["trait"] = traits,
                              Rcpp::_["level"] = levels,
                              Rcpp::_["alpha"] = alpha_list,
                              Rcpp::_["mean"] = mean_list,
                              Rcpp::_["sd"] = sd_list,
                              Rcpp::_["lbf"] = lbf_list,
                              Rcpp::_["z"] = z_list,
                              Rcpp::_["lfsr"] = lfsr_list,
                              Rcpp::_["lodds"] = lodds_list);
}

#endif
