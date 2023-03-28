#ifndef CFSUSIE_REGRESSION_HH_
#define CFSUSIE_REGRESSION_HH_

template <typename MODEL, typename STAT, typename Derived>
Scalar
update_paired_regression(MODEL &model_fa, // factual model
                         MODEL &model_cf, // counterfactual model
                         STAT &stat_fa,   // factual stat
                         STAT &stat_cf,   // counterfactual stat
                         const Eigen::MatrixBase<Derived> &X_fa,
                         const Eigen::MatrixBase<Derived> &Y_fa,
                         const Eigen::MatrixBase<Derived> &X_cf,
                         const Eigen::MatrixBase<Derived> &Y_cf,
                         const Scalar lodds_cutoff = 0)
{
    Scalar llik = 0, llik1, llik0;

    const Index L = std::min(model_fa.lvl, model_cf.lvl);

    for (Index l = 0; l < L; ++l) {

        discount_model_stat(model_fa, X_fa, Y_fa, l); //
        discount_model_stat(model_cf, X_cf, Y_cf, l); //

        llik1 = SER(X_fa,                // Single-effect regression
                    model_fa.partial_nm, //   - partial prediction
                    model_fa.residvar_m, //   - residual variance
                    model_fa.get_v0(l),  //   - prior variance
                    stat_fa,             //   - statistics
                    lodds_cutoff);       //   - log-odds cutoff

        llik0 = SER(X_cf,                // Single-effect regression
                    model_cf.partial_nm, //   - partial prediction
                    model_cf.residvar_m, //   - residual variance
                    model_cf.get_v0(l),  //   - prior variance
                    stat_cf,             //   - statistics
                    lodds_cutoff);       //   - log-odds cutoff

        // Recalibrate posterior statistics and prior vars
        calibrate_prior_var(stat_fa);
        calibrate_prior_var(stat_cf);
        calibrate_post_stat(stat_fa, stat_fa.v0);
        calibrate_post_stat(stat_cf, stat_cf.v0);

        update_model_stat(model_fa, X_fa, stat_fa, l);
        update_model_stat(model_cf, X_cf, stat_cf, l);

        llik += llik1;
    }

    calibrate_residual_variance(model_fa, X_fa, Y_fa);
    calibrate_residual_variance(model_cf, X_cf, Y_cf);

    return llik;
}

#endif
