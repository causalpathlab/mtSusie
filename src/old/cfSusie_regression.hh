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

    const Index L = std::min(model_fa.lvl, model_cf.lvl);

    Scalar score = 0.;

    for (Index l = 0; l < L; ++l) {

        discount_model_stat(model_fa, X_fa, Y_fa, l); //
        discount_model_stat(model_cf, X_cf, Y_cf, l); //

        score += SER(X_fa,                // Single-effect regression
                     model_fa.partial_nm, //   - partial prediction
                     model_fa.residvar_m, //   - residual variance
                     model_fa.get_v0(l),  //   - prior variance
                     stat_fa,             //   - statistics
                     lodds_cutoff);       //   - log-odds cutoff

        score += SER(X_cf,                // Single-effect regression
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
    }

    calibrate_residual_variance(model_fa, X_fa, Y_fa);
    calibrate_residual_variance(model_cf, X_cf, Y_cf);

    return score;
}

#endif
