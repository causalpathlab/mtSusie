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
        , alpha0_p(p)
        , lbf_p(p)
        , lbf0_p(p)
        , post_mean_pm(p, m)
        , post_var_pm(p, m)
        , lbf_pm(p, m)
        , mle_mean_pm(p, m)
        , mle_var_pm(p, m)
        , XtY_pm(p, m)
        , x2_p(p)
        , lodds_m(m)
        , temp_m(m)
        , v0(1)
        , lbf_op(v0)
    {
        alpha_p.setZero();
        alpha0_p.setZero();
        lbf_p.setZero();
        lbf0_p.setZero();
        mle_mean_pm.setZero();
        mle_var_pm.setOnes();
    }

    const Index p, m;

    ColVec alpha_p;   // p x 1 combined PIP
    ColVec alpha0_p;  // p x 1 null PIP
    ColVec lbf_p;     // log Bayes Factor
    ColVec lbf0_p;    // null lbf
    Mat post_mean_pm; // posterior mean
    Mat post_var_pm;  // posterior variance

    Mat lbf_pm;      // log bayes factor
    Mat mle_mean_pm; // MLE mean
    Mat mle_var_pm;  // MLE variance

    Mat XtY_pm;     // sufficient stat
    ColVec x2_p;    // sufficient stat
    RowVec lodds_m; // 1 x m log odds
    RowVec temp_m;  // 1 x m temporary

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
            Scalar stuff1 = std::log1p(v0 / s2);
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
calibrate_prior_var(STAT &stat, const Scalar eps = 1e-8)
{
    stat.v0 = sum_safe(
        (stat.post_mean_pm.cwiseProduct(stat.post_mean_pm) + stat.post_var_pm)
            .transpose() *
        stat.alpha_p.transpose());
    stat.v0 += eps;
}

template <typename STAT>
Index
calibrate_post_selection(STAT &stat, const Scalar lodds_cutoff)
{
    const Index m = stat.lodds_m.size();
    Index nretain = m, nretain_old = m;
    const Scalar p0 = 1. / static_cast<Scalar>(stat.p);
    stat.alpha0_p.setConstant(p0);

    // a. Initialization of shared PIP
    {
        stat.lbf_p = stat.lbf_pm.rowwise().sum();
        Scalar maxlbf = stat.lbf_p.maxCoeff();
        stat.alpha_p = (stat.lbf_p.array() - maxlbf).exp();
        stat.alpha_p /= stat.alpha_p.sum();
    }

    if (stat.m == 1) { // single output
        stat.lodds_m = (stat.alpha_p - stat.alpha0_p).transpose() * stat.lbf_pm;
        return nretain; // nothing to do
    }

    for (Index inner = 0; inner < m; ++inner) {
        // b. Determine which outputs to include
        // H1: sum_j log alpha[j,k] * alpha[j]
        // H0: sum_j log alpha[j,k] * null[j]
        stat.lodds_m = (stat.alpha_p - stat.alpha0_p).transpose() * stat.lbf_pm;
        nretain = (stat.lodds_m.array() > lodds_cutoff).count();

        // c. only some of the output variables > lodds
        if (nretain > 0 && nretain < nretain_old) {
            stat.lbf_p.setZero();
            stat.lbf0_p.setZero();
            for (Index k = 0; k < m; ++k) {
                if (stat.lodds_m(k) > lodds_cutoff) {
                    stat.lbf_p += stat.lbf_pm.col(k);
                } else {
                    stat.lbf0_p += stat.lbf_pm.col(k);
                }
            }

            {
                const Scalar maxlbf = stat.lbf_p.maxCoeff();
                stat.alpha_p = (stat.lbf_p.array() - maxlbf).exp();
                stat.alpha_p /= stat.alpha_p.sum();
            }

            {
                const Scalar maxlbf0 = stat.lbf0_p.maxCoeff();
                stat.alpha0_p = (stat.lbf0_p.array() - maxlbf0).exp();
                stat.alpha0_p /= stat.alpha0_p.sum();
            }

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
    // N(0| b, v1 + v0) / N(0| b, v0)
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
                const Eigen::MatrixBase<Derived3> &resid_var_m,
                const Scalar eps = 1e-8)
{
    XtY_safe(X, Y, stat.XtY_pm);                           // p x m
    rowsum_safe(X.cwiseProduct(X).transpose(), stat.x2_p); // p x 1

    // b[j,k] = sum_i x[i,j] y[i,k] / sum_i x[i,j]^2  (p x m)
    stat.mle_mean_pm = stat.XtY_pm.array().colwise() / stat.x2_p.array();

    // s2[j,k] = sigma[k]^2 / sum_i x[i,j]^2          (p x m)
    stat.mle_var_pm.setZero();
    stat.mle_var_pm.array().rowwise() += resid_var_m.row(0).array();
    stat.mle_var_pm.array().colwise() /= stat.x2_p.array();
    stat.mle_var_pm.array() += eps;
}

template <typename STAT>
void
calibrate_post_stat(STAT &stat, const Scalar v0, const Scalar eps = 1e-8)
{
    stat.post_var_pm =
        (stat.mle_var_pm.array().inverse() + 1. / v0).inverse() + eps;

    stat.post_mean_pm = stat.mle_mean_pm.cwiseProduct(stat.post_var_pm)
                            .cwiseQuotient(stat.mle_var_pm);
}

template <typename STAT,
          typename Derived1,
          typename Derived2,
          typename Derived3>
Scalar
calculate_posterior_loglik(STAT &S,
                           const Eigen::MatrixBase<Derived1> &X,
                           const Eigen::MatrixBase<Derived2> &Y,
                           const Eigen::MatrixBase<Derived3> &resid_var_m)
{
    Scalar llik = 0;
    const Scalar n = X.rows();

    // 1. Square terms
    //    -0.5 * sum((Y - X * (mu * alpha))^2 / vR)

    llik -= 0.5 *
        sum_safe(((Y -
                   X *
                       (S.post_mean_pm.array().colwise() * S.alpha_p.array())
                           .matrix())
                      .pow(2.)
                      .array()
                      .rowwise() /
                  resid_var_m.array())
                     .matrix());

    llik -=
        0.5 *
        sum_safe(
            ((X.cwiseProduct(X) *
              (S.post_var_pm.array().colwise() * S.alpha_p.array()).matrix())
                 .array()
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
SER(const Eigen::MatrixBase<Derived1> &X,
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
