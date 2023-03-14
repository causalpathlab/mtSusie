#include "mtSusie.hh"

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
        , resid_var_lm(levels, num_output)
        , partial_nm(num_sample, num_output)
        , prior_var(levels)
        , temp_m(num_output)
    {
        shared_pip_pl.setZero();
        fitted_nm.setZero();
        resid_var_lm.setOnes();
        prior_var.setConstant(vv);

        fill_mat_vec(mu_pm_list, lvl, p, m);
        fill_mat_vec(var_pm_list, lvl, p, m);
        fill_mat_vec(z_pm_list, lvl, p, m);
        fill_mat_vec(lbf_pm_list, lvl, p, m);
    }

    const Index n, m, p, lvl;

    Mat shared_pip_pl;
    Mat fitted_nm;
    Mat resid_var_lm;
    Mat partial_nm;
    RowVec prior_var;
    RowVec temp_m;

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

    Scalar get_v0(Index l) const
    {
        check_lvl(l);
        return prior_var(l);
    }

private:
    inline Mat &_get(mat_vec_t &vec, Index l)
    {
        check_lvl(l);
        return vec[l];
    }
    template <typename Derived>
    inline void
    _set(mat_vec_t &vec, Index l, const Eigen::MatrixBase<Derived> &x)
    {
        check_lvl(l);
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

template <typename MODEL, typename STAT, typename Derived1, typename Derived2>
Scalar
update_shared_effects(MODEL &model,
                      STAT &stat,
                      const Eigen::MatrixBase<Derived1> &X,
                      const Eigen::MatrixBase<Derived2> &Y)
{
    Scalar llik = 0;
    const Index L = model.lvl;

    for (Index l = 0; l < L; ++l) {

        // 1. Remove l-th effect from the fitted values
        XY_safe(X,
                (model.get_mean(l).array().colwise() *
                 model.shared_pip_pl.col(l).array())
                    .matrix(),
                model.partial_nm);

        model.fitted_nm -= model.partial_nm;

        // 2. Take partial residuals
        model.partial_nm = Y - model.fitted_nm;
        const Scalar v0 = model.get_v0(l);

        // 3. Update by shared "single" effect regression
        colsum_safe(model.partial_nm.cwiseProduct(model.partial_nm),
                    model.temp_m);

        const Scalar nn = static_cast<Scalar>(model.n);
        model.resid_var_lm.row(l) = model.temp_m / nn;

        Scalar llik_l = fit_single_effect_shared(X,
                                                 model.partial_nm,
                                                 model.resid_var_lm.row(l),
                                                 v0,
                                                 stat);

        // 4. Put back the updated statistics
        model.shared_pip_pl.col(l) = stat.alpha_p;
        model.set_mean(l, stat.post_mean_pm);
        model.set_var(l, stat.post_var_pm);
        model.set_lbf(l, stat.lbf_pm);
        model.prior_var(l) = stat.v0;
        model.set_z(l,
                    stat.mle_mean_pm.cwiseQuotient(
                        stat.mle_var_pm.cwiseSqrt()));

        XY_safe(X,
                (model.get_mean(l).array().colwise() *
                 model.shared_pip_pl.col(l).array())
                    .matrix(),
                model.partial_nm);

        model.fitted_nm += model.partial_nm;

        llik += llik_l;
        // TLOG("level: [" << l << "] " << llik_l);
    }

    return llik;
}

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
                              Rcpp::_["z"] = model.z_pm_list);
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
             const double tol = 1e-8,
             const double prior_var = 100.0)
{

    shared_regression_t model(y.rows(), y.cols(), x.cols(), levels, prior_var);
    shared_effect_stat_t stat(x.cols(), y.cols());

    std::vector<Scalar> loglik;
    loglik.reserve(max_iter);

    const Mat xx = Rcpp::as<Mat>(x), yy = Rcpp::as<Mat>(y);

    for (Index iter = 0; iter < max_iter; ++iter) {

        const Scalar curr = update_shared_effects(model, stat, xx, yy);

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
    return ret;
}
