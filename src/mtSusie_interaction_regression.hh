#ifndef MTSUSIE_INTERACTION_REGRESSION_HH_
#define MTSUSIE_INTERACTION_REGRESSION_HH_

template <typename MODEL, typename STAT, typename Derived>
Scalar
update_shared_interaction_regression(
    MODEL &model,
    STAT &stat,
    const Eigen::MatrixBase<Derived> &X, // design matrix
    const Eigen::MatrixBase<Derived> &Y, // output
    const Eigen::MatrixBase<Derived> &W, // interaction terms
    const std::vector<int> &w_terms,     // 1:L W column indexes
    const std::vector<bool> &x_terms,     // 1:L true or false
    const bool local_residual = false,
    const bool do_stdize_lbf = false,
    const bool do_update_prior = false,
    const bool do_hard_selection = false,
    const Scalar hard_lodds_cutoff = 0)

{

    Mat WX(X.rows(), X.cols()); // temporary w[i,k] * x[i,j]
    const Index L = model.lvl;

    Scalar score = 0.;

    for (Index l = 0; l < L; ++l) {

        //////////////////////////////////////////
        // pick interaction and predictor terms //
        //////////////////////////////////////////

        const Index x_k = x_terms.at(l);
        const Index w_k = w_terms.at(l);

	WX.setZero();

	if(x_k >= 0 && w_k >=0){
	  WX.resize(X.rows(), X.cols());

        //     WX = X.array().colwise() * W.col(k).array(); // weighted by W[,k]

	} else if(w_k >= 0 && x_k < 0){

	} else {
	  WX.resize(X.rows(), X.cols());
	  WX = X;
	}


        // const Index k = interaction.at(l);
        // if (k >= 0) {
        //     WX = X.array().colwise() * W.col(k).array(); // weighted by W[,k]
        // } else {                                         //
        //     WX = X;                                      // unweighted
        // }

        discount_model_stat(model, WX, Y, l); // 1. discount previous l-th

        if (local_residual) {
            calibrate_residual_variance(model, model.partial_nm);
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
        calibrate_residual_variance(model, Y);
    }

    return score;
}

#endif
