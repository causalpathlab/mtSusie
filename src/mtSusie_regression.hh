#ifndef MTSUSIE_REGRESSION_HH_
#define MTSUSIE_REGRESSION_HH_

struct shared_regression_t {

    using mat_vec_t = std::vector<Mat>;

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    explicit shared_regression_t(const Index num_sample,
                                 const Index num_output,
                                 const Index num_feature,
                                 const Index levels,
                                 const Scalar vv)
        : n(num_sample)
        , m(num_output)
        , p(num_feature)
        , lvl(levels)
        , shared_pip_pl(num_feature, levels)
        , fitted_nm(num_sample, num_output)
        , residvar_m(num_output)
        , partial_nm(num_sample, num_output)
        , residuals_nm(num_sample, num_output)
        , prior_var(levels)
        , temp_m(1, num_output)
        , temp2_m(1, num_output)
        , lodds_lm(levels, num_output)
    {
        shared_pip_pl.setConstant(1. / static_cast<Scalar>(p));
        fitted_nm.setZero();
        residvar_m.setOnes();
        prior_var.setConstant(vv);

        fill_mat_vec(mu_pm_list, lvl, p, m);
        fill_mat_vec(var_pm_list, lvl, p, m);
        fill_mat_vec(z_pm_list, lvl, p, m);
        fill_mat_vec(lbf_pm_list, lvl, p, m);

        for (std::size_t l = 0; l < lvl; ++l) {
            _get(mu_pm_list, l).setZero();
            _get(z_pm_list, l).setZero();
            _get(lbf_pm_list, l).setZero();
            _get(var_pm_list, l).setOnes();
        }
    }

    const Index n, m, p, lvl;

    Mat shared_pip_pl;
    Mat fitted_nm;
    RowVec residvar_m;
    Mat partial_nm;
    Mat residuals_nm;
    RowVec prior_var;
    RowVec temp_m;
    RowVec temp2_m;
    Mat lodds_lm;

    mat_vec_t mu_pm_list;
    mat_vec_t var_pm_list;
    mat_vec_t z_pm_list;
    mat_vec_t lbf_pm_list;

    Mat &get_mean(Index l) { return _get(mu_pm_list, l); }
    Mat &get_var(Index l) { return _get(var_pm_list, l); }
    Mat &get_z(Index l) { return _get(z_pm_list, l); }
    Mat &get_lbf(Index l) { return _get(lbf_pm_list, l); }

    template <typename Derived>
    void set_mean(Index l, const Eigen::MatrixBase<Derived> &x)
    {
        _set(mu_pm_list, l, x);
    }
    template <typename Derived>
    void set_var(Index l, const Eigen::MatrixBase<Derived> &x)
    {
        _set(var_pm_list, l, x);
    }

    template <typename Derived>
    void set_z(Index l, const Eigen::MatrixBase<Derived> &x)
    {
        _set(z_pm_list, l, x);
    }

    template <typename Derived>
    void set_lbf(Index l, const Eigen::MatrixBase<Derived> &x)
    {
        _set(lbf_pm_list, l, x);
    }

    Scalar get_v0(Index l)
    {
#ifdef DEBUG
        check_lvl(l);
#endif
        return prior_var(l);
    }

private:
    inline Mat &_get(mat_vec_t &vec, Index l)
    {
#ifdef DEBUG
        check_lvl(l);
#endif
        return vec.at(l);
    }
    template <typename Derived>
    inline void
    _set(mat_vec_t &vec, Index l, const Eigen::MatrixBase<Derived> &x)
    {
#ifdef DEBUG
        check_lvl(l);
#endif
        vec[l].setZero();
        vec[l] += x;
    }

    void check_lvl(Index l) const
    {
        ASSERT(l >= 0 && l < lvl, "check the index l: " << l);
    }

    void fill_mat_vec(mat_vec_t &mat_vec,
                      const Index size,
                      const Index d1,
                      const Index d2)
    {
        mat_vec.clear();
        mat_vec.reserve(size);
        for (Index k = 0; k < size; ++k) {
            mat_vec.emplace_back(Mat(d1, d2));
        }
    }
};

template <typename MODEL, typename Derived, typename Derived2>
void
discount_model_stat(MODEL &model,
                    const Eigen::MatrixBase<Derived> &X,
                    const Eigen::MatrixBase<Derived2> &Y,
                    const Index l)
{
    XY_safe(X,
            (model.get_mean(l).array().colwise() *
             model.shared_pip_pl.col(l).array())
                .matrix(),
            model.partial_nm);

    model.fitted_nm -= model.partial_nm;

    // Take partial residuals
    model.partial_nm = Y - model.fitted_nm;
}

template <typename MODEL, typename Derived>
Mat
model_residuals(MODEL &model, const Eigen::MatrixBase<Derived> &Y)
{
    model.residuals_nm = Y - model.fitted_nm;
    return model.residuals_nm;
}

template <typename MODEL>
Mat
model_fitted(MODEL &model)
{
    return model.fitted_nm;
}

template <typename MODEL, typename Derived>
void
calibrate_residual_variance(MODEL &model,
                            const Eigen::MatrixBase<Derived> &X,
                            const Eigen::MatrixBase<Derived> &Y)
{

    // Expected R2 value for each output
    // (Y - X * E[theta])^2
    const Scalar nn = X.rows();
    colsum_safe((Y - model.fitted_nm).cwiseProduct(Y - model.fitted_nm),
                model.residvar_m);

    // Exact calculation of X^2 * V[theta] easily blow up!!
    model.residvar_m /= nn;
}

template <typename MODEL, typename Derived, typename STAT>
void
update_model_stat(MODEL &model,
                  const Eigen::MatrixBase<Derived> &X,
                  STAT &stat,
                  const Index l)
{
    model.shared_pip_pl.col(l) = stat.alpha_p;
    model.set_mean(l, stat.post_mean_pm);
    model.set_var(l, stat.post_var_pm);
    model.set_lbf(l, stat.lbf_pm);
    model.prior_var(l) = stat.v0;

    model.set_z(l, stat.mle_mean_pm.cwiseQuotient(stat.mle_var_pm.cwiseSqrt()));

    model.lodds_lm.row(l) = stat.lodds_m;

    XY_safe(X,
            (model.get_mean(l).array().colwise() *
             model.shared_pip_pl.col(l).array())
                .matrix(),
            model.partial_nm);

    model.fitted_nm += model.partial_nm;
}

template <typename MODEL>
Scalar
recon_error(MODEL &model)
{
    return model.residvar_m.sum();
}

template <typename MODEL, typename STAT, typename Derived>
Scalar
update_shared_regression(MODEL &model,
                         STAT &stat,
                         const Eigen::MatrixBase<Derived> &X,
                         const Eigen::MatrixBase<Derived> &Y,
                         const bool local_residual = false,
                         const bool do_stdize_lbf = false,
                         const bool do_update_prior = false,
                         const bool do_hard_selection = false,
                         const Scalar hard_lodds_cutoff = 0)
{
    const Index L = model.lvl;
    Scalar score = 0.;

    for (Index l = 0; l < L; ++l) {

        // 1. discount previous l-th
        discount_model_stat(model, X, Y, l);

        // 2. calibrate residual variance locally...
        if (local_residual) {
            calibrate_residual_variance(model, X, model.partial_nm);
        }

        score += SER(X,                  // 3. single-effect regression
                     model.partial_nm,   //   - partial prediction
                     model.residvar_m,   //   - residual variance
                     model.get_v0(l),    //   - prior variance
                     stat,               //   - statistics
                     do_stdize_lbf,      //
                     do_update_prior,    //
                     do_hard_selection,  //
                     hard_lodds_cutoff); //

        update_model_stat(model, X, stat, l); // Put back the updated stats
    }

    // Calibrate residual calculation globally
    if (!local_residual) {
        calibrate_residual_variance(model, X, Y);
    }

    return score;
}

#endif
