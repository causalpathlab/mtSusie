#ifndef MTSUSIE_INTERACTION_REGRESSION_HH_
#define MTSUSIE_INTERACTION_REGRESSION_HH_

template <typename MODEL,
          typename STAT,
          typename Derived1,
          typename Derived2,
          typename Derived3>
Scalar
update_shared_interaction_regression(MODEL &model,
                                     STAT &stat,
                                     const Eigen::MatrixBase<Derived1> &X,
                                     const Eigen::MatrixBase<Derived2> &Y,
                                     const Eigen::MatrixBase<Derived3> &W,
                                     const std::vector<int> &interaction,
                                     const Index levels_per_inter,
                                     const Scalar lodds_cutoff = 0)

{
    Scalar llik = 0, llik_;

    Mat WX(X.rows(), X.cols()); // w[i,k] * x[i,j]
    const Index L = model.lvl;
    const Index K = W.cols();
    for (Index l = 0; l < L; ++l) {
        const Index k = interaction.at(l);
        if (k >= 0) {
            WX = X.array().colwise() * W.col(k).array(); // weighted by W[,k]
        } else {                                         //
            WX = X;                                      // unweighted
        }

        discount_model_stat(model, WX, Y, l);  // 1. discount previous l-th
        calibrate_residual_variance(model, l); // 2. calibrate variance
        llik_ = SER(WX,                        // 3. single-effect regression
                    model.partial_nm,          //   - partial prediction
                    model.residvar_lm.row(l),  //   - residual variance
                    model.get_v0(l),           //   - prior variance
                    stat,                      //   - statistics
                    lodds_cutoff);             //   - log-odds cutoff
        update_model_stat(model, WX, stat, l); // Put back the updated results
        llik += llik_;                         // log-likelihood
    }

    return llik;
}

#endif
