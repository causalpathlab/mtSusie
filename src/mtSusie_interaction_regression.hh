#ifndef MTSUSIE_INTERACTION_REGRESSION_HH_
#define MTSUSIE_INTERACTION_REGRESSION_HH_

template <typename MODEL, typename Derived>
void
calibrate_residual_variance(MODEL &model,
                            const Eigen::MatrixBase<Derived> &X,
                            const Eigen::MatrixBase<Derived> &Y,
                            const Eigen::MatrixBase<Derived> &W,
                            const std::vector<int> &interaction)
{

    const Index L = model.lvl;
    const Index K = W.cols();
    const Scalar nn = X.rows();

    // Expected R2 value for each output
    // (Y - X * E[theta])^2
    colsum_safe((Y - model.fitted_nm).cwiseProduct(Y - model.fitted_nm),
                model.residvar_m);

    // Exact calculation of X^2 * V[theta] easily blow up!!
    model.residvar_m /= nn;
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
                                     const bool local_residual = false,
                                     const bool do_stdize_lbf = false,
                                     const bool do_update_prior = false,
                                     const bool do_hard_selection = false,
                                     const Scalar hard_lodds_cutoff = 0)

{

    Mat WX(X.rows(), X.cols()); // w[i,k] * x[i,j]
    const Index L = model.lvl;
    const Index K = W.cols();

    Scalar score = 0.;

    for (Index l = 0; l < L; ++l) {

        ///////////////////////////
        // pick interaction term //
        ///////////////////////////

        const Index k = interaction.at(l);
        if (k >= 0) {
            WX = X.array().colwise() * W.col(k).array(); // weighted by W[,k]
        } else {                                         //
            WX = X;                                      // unweighted
        }

        discount_model_stat(model, WX, Y, l); // 1. discount previous l-th

        if (local_residual) {
            calibrate_residual_variance(model,
                                        X,
                                        model.partial_nm,
                                        W,
                                        interaction);
        }

        score += SER(WX,                 // 3. single-effect regr
                     model.partial_nm,   //   - partial prediction
                     model.residvar_m,   //   - residual variance
                     model.get_v0(l),    //   - prior variance
                     stat,               //   - statistics
                     do_stdize_lbf,      //
                     do_update_prior,    //
                     do_hard_selection,  //
                     hard_lodds_cutoff); //

        update_model_stat(model, WX, stat, l); // Put back the updated stat
    }

    // Calibrate residual calculation globally
    if (!local_residual) {
        calibrate_residual_variance(model, X, Y, W, interaction);
    }

    return score;
}

#endif
