#ifndef MTSUSIE_SINGLE_EFFECT_HH_
#define MTSUSIE_SINGLE_EFFECT_HH_

struct shared_effect_stat_t {

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    explicit shared_effect_stat_t(const Index num_variable,
                                  const Index num_output)
        : p(num_variable)
        , m(num_output)
        , alpha_p(p)
        , lbf_p(p)
        , post_mean_pm(p, m)
        , post_var_pm(p, m)
        , lbf_pm(p, m)
        , mle_mean_pm(p, m)
        , mle_var_pm(p, m)
        , prior_var_m(m)
        , XtY_pm(p, m)
        , x2_p(p)
        , lodds_m(m)
        , v0(1)
        , lbf_op(v0)
    {
    }

    const Index p, m;

    ColVec alpha_p;   // p x 1 combined PIP
    ColVec lbf_p;     // log Bayes Factor
    Mat post_mean_pm; // posterior mean
    Mat post_var_pm;  // posterior variance

    Mat lbf_pm;         // log bayes factor
    Mat mle_mean_pm;    // MLE mean
    Mat mle_var_pm;     // MLE variance
    RowVec prior_var_m; // prior variance

    Mat XtY_pm;     // sufficient stat
    ColVec x2_p;    // sufficient stat
    RowVec lodds_m; // 1 x m log odds

    Scalar v0;

    // logN(b, 0, v + s2) = - 0.5 log(s2 + v) - 0.5 b^2 / (v + s2)
    // logN(b, 0, s2) = - 0.5 log(s2) - 0.5 b^2 / s2
    //
    // logBF = -0.5 log(1 + v/s2) + 0.5 b^2 (1/s2 - 1/(s2 + v))
    struct lbf_op_t {
        explicit lbf_op_t(const Scalar &v)
            : v0(v)
        {
        }
        Scalar operator()(const Scalar &b, const Scalar &s2) const
        {
            if (!std::isfinite(s2) || s2 <= tol) { // ignore infinte s2
                return 0;                          // also zero s2
            }
            Scalar stuff1 = fasterlog(1. + v0 / s2);
            Scalar stuff2 = b * b / s2 - b * b / (s2 + v0);
            return 0.5 * (stuff2 - stuff1);
        }
        const Scalar &v0;
        static constexpr Scalar tol = 1e-8;
    } lbf_op;
};

template <typename STAT>
void
set_prior_var(STAT &stat, const Scalar v0)
{
    stat.v0 = v0;
}

template <typename STAT>
void
calibrate_prior_var(STAT &stat)
{
    stat.v0 = sum_safe(
        stat.alpha_p.transpose() *
        (stat.post_mean_pm.cwiseProduct(stat.post_mean_pm) + stat.post_var_pm));
}

template <typename STAT>
Index
calibrate_post_selection(STAT &stat, const Scalar lodds_cutoff)
{
    {
        stat.lbf_p = stat.lbf_pm.rowwise().sum();
        const Scalar maxlbf = stat.lbf_p.maxCoeff();
        stat.alpha_p = (stat.lbf_p.array() - maxlbf).exp();
        stat.alpha_p /= stat.alpha_p.sum();
    }
    const Index m = stat.lodds_m.size();
    Index nretain = m, nretain_old = m;

    for (Index inner = 0; inner < m; ++inner) {
        // a. Determine which outputs to include
        // H1: sum_j log alpha[j,k] * alpha[j]
        // H0: sum_j log alpha[j,k] * 1/p
        stat.lodds_m = stat.alpha_p.transpose() * stat.lbf_pm;
        stat.lodds_m -= stat.lbf_pm.colwise().mean();

        nretain = (stat.lodds_m.array() > lodds_cutoff).count();

        // b. only some of the output variables > lodds
        if (nretain > 0 && nretain < nretain_old) {
            stat.lbf_p.setZero();
            for (Index k = 0; k < m; ++k) {
                if (stat.lodds_m(k) > lodds_cutoff) {
                    stat.lbf_p += stat.lbf_pm.col(k);
                }
            }

            const Scalar maxlbf = stat.lbf_p.maxCoeff();
            stat.alpha_p = (stat.lbf_p.array() - maxlbf).exp();
            stat.alpha_p /= stat.alpha_p.sum();
            nretain_old = nretain;
        } else {
            break;
        }
    }
    return nretain;
}

template <typename STAT>
void
calibrate_lbf(STAT &stat)
{
    stat.lbf_pm = stat.mle_mean_pm.binaryExpr(stat.mle_var_pm, stat.lbf_op);
}

template <typename STAT,
          typename Derived1,
          typename Derived2,
          typename Derived3>
void
update_mle_stat(STAT &stat,
                const Eigen::MatrixBase<Derived1> &X,
                const Eigen::MatrixBase<Derived2> &Y,
                const Eigen::MatrixBase<Derived3> &resid_var_m)
{
    XtY_safe(X, Y, stat.XtY_pm);                           // p x m
    rowsum_safe(X.cwiseProduct(X).transpose(), stat.x2_p); // p x 1

    // b[j,k] = sum_i x[i,j] y[i,k] / sum_i x[i,j]^2  (p x m)
    stat.mle_mean_pm = stat.XtY_pm.array().colwise() / stat.x2_p.array();

    // s2[j,k] = sigma[k]^2 / sum_i x[i,j]^2          (p x m)
    stat.mle_var_pm.setZero();
    stat.mle_var_pm.rowwise() += resid_var_m;
    stat.mle_var_pm.array().colwise() /= stat.x2_p.array();
}

template <typename STAT>
void
calibrate_post_stat(STAT &stat, const Scalar v0)
{
    stat.post_var_pm = (stat.mle_var_pm.array().inverse() + 1. / v0).inverse();
    stat.post_mean_pm = stat.mle_mean_pm.cwiseProduct(
        stat.post_var_pm.cwiseQuotient(stat.mle_var_pm));
}

template <typename STAT,
          typename Derived1,
          typename Derived2,
          typename Derived3>
Scalar
calculate_posterior_loglik(STAT &stat,
                           const Eigen::MatrixBase<Derived1> &X,
                           const Eigen::MatrixBase<Derived2> &Y,
                           const Eigen::MatrixBase<Derived3> &resid_var_m)
{
    Scalar llik = 0;
    const Scalar n = X.rows();

    // hat <- safe.xty(t(X), sweep(mu, 1, alpha, `*`))
    // hat2 <- safe.xty(t(X * X), sweep(mu2, 1, alpha, `*`))
    // stuff <- apply(Y*Y - 2*Y*hat + hat2, 2, sum, na.rm=TRUE)
    //  - 0.5 * sum(stuff / vR, na.rm=TRUE)

    // 1. Square terms
    //    -0.5 * sum(Y^2 / vR)
    //    Y .* (X * (mu .* alpha))
    //    - 0.5 * X^2 * ((mu^2 + var) .* alpha)

    llik -= 0.5 *
        sum_safe((Y.array().pow(2.).rowwise() / resid_var_m.array()).matrix());

    llik += sum_safe(Y.cwiseProduct(
        X *
        ((stat.post_mean_pm.array().colwise() * stat.alpha_p.array())
             .rowwise() /
         resid_var_m.array())
            .matrix()));

    llik -=
        0.5 *
        sum_safe(
            X.cwiseProduct(X) *
            (((stat.post_mean_pm.pow(2.) + stat.post_var_pm).array().colwise() *
              stat.alpha_p.array())
                 .rowwise() /
             resid_var_m.array())
                .matrix());

    // 2. Log terms
    //    -0.5 * n * sum(log(2*pi*vR)
    llik -=
        0.5 * n * sum_safe((resid_var_m * 2. * M_PI).array().log().matrix());

    return llik;
}

template <typename Derived1,
          typename Derived2,
          typename Derived3,
          typename STAT>
Scalar
fit_single_effect_shared(const Eigen::MatrixBase<Derived1> &X,
                         const Eigen::MatrixBase<Derived2> &Y,
                         const Eigen::MatrixBase<Derived3> &resid_var_m,
                         const Scalar v0,
                         STAT &stat,
                         const Scalar lodds_cutoff = 0)
{
    set_prior_var(stat, v0);

    // 1. Update sufficient statistics
    update_mle_stat(stat, X, Y, resid_var_m);
    calibrate_lbf(stat);

    // 2. Refine joint variable selection
    const Index k = calibrate_post_selection(stat, lodds_cutoff);
    // TLOG(k << " selected");

    // 3. Calibrate posterior statistics
    calibrate_post_stat(stat, v0);
    calibrate_prior_var(stat);

    return calculate_posterior_loglik(stat, X, Y, resid_var_m);
}

#endif