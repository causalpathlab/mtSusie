#ifndef MTSUSIE_INTERACTION_REGRESSION_HH_
#define MTSUSIE_INTERACTION_REGRESSION_HH_

template <typename MODEL, typename Derived>
void
calibrate_residual_variance(MODEL &model,
                            const Eigen::MatrixBase<Derived> &X,
                            const Eigen::MatrixBase<Derived> &Y,
                            const Eigen::MatrixBase<Derived> &W,
                            const std::vector<int> &interaction,
                            const Scalar eps = 1e-4)
{

    Mat WX(X.rows(), X.cols()); // w[i,k] * x[i,j]

    const Index L = model.lvl;
    const Index K = W.cols();
    const Scalar nn = X.rows();

    // Expected R2 value for each output
    // (Y - X * E[theta])^2
    colsum_safe((Y - model.fitted_nm).cwiseProduct(Y - model.fitted_nm),
                model.residvar_m);

    // X^2 * V[theta]
    for (Index l = 0; l < L; ++l) {
        const Mat &var = model.get_var(l); // p x m

        const Index k = interaction.at(l);
        if (k >= 0) {
            colsum_safe(((X.array().colwise() * W.col(k).array()) *
                         (X.array().colwise() * W.col(k).array()))
                                .matrix() *
                            ((var.cwiseProduct(var)).array() *
                             model.shared_pip_pl.array().col(l))
                                .matrix(),
                        model.temp_m);

        } else { //
            colsum_safe(X.cwiseProduct(X) *
                            ((var.cwiseProduct(var)).array() *
                             model.shared_pip_pl.array().col(l))
                                .matrix(),
                        model.temp_m);
        }

        model.residvar_m += model.temp_m;
    }

    model.residvar_m = model.residvar_m / nn;
    model.residvar_m.array() += eps;
}

template <typename MODEL, typename STAT, typename Derived>
Scalar
update_shared_interaction_regression(MODEL &model,
                                     STAT &stat,
                                     const Eigen::MatrixBase<Derived> &X,
                                     const Eigen::MatrixBase<Derived> &Y,
                                     const Eigen::MatrixBase<Derived> &W,
                                     const std::vector<int> &interaction,
                                     const Index levels_per_inter,
                                     const Scalar lodds_cutoff = 0)

{

    Mat WX(X.rows(), X.cols()); // w[i,k] * x[i,j]
    const Index L = model.lvl;
    const Index K = W.cols();

    Scalar score = 0.;

    for (Index l = 0; l < L; ++l) {
        const Index k = interaction.at(l);
        if (k >= 0) {
            WX = X.array().colwise() * W.col(k).array(); // weighted by W[,k]
        } else {                                         //
            WX = X;                                      // unweighted
        }

        discount_model_stat(model, WX, Y, l);  // 1. discount previous l-th
        score += SER(WX,                       // 2. single-effect regr
                     model.partial_nm,         //   - partial prediction
                     model.residvar_m,         //   - residual variance
                     model.get_v0(l),          //   - prior variance
                     stat,                     //   - statistics
                     lodds_cutoff);            //   - log-odds cutoff
        update_model_stat(model, WX, stat, l); // Put back the updated stat
    }

    calibrate_residual_variance(model, X, Y, W, interaction);

    return score;
}

#endif
