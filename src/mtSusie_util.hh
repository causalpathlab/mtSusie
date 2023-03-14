#ifndef MTSUSIE_UTIL_HH_
#define MTSUSIE_UTIL_HH_

template <typename T, typename Ret>
void
colsum_safe(const Eigen::MatrixBase<T> &X, Eigen::MatrixBase<Ret> &ret)
{
    inf_zero_op<Mat> op;
    ret.derived() = X.unaryExpr(op).colwise().sum();
}

template <typename T, typename Ret>
void
rowsum_safe(const Eigen::MatrixBase<T> &X, Eigen::MatrixBase<Ret> &ret)
{
    inf_zero_op<Mat> op;
    ret.derived() = X.unaryExpr(op).rowwise().sum();
}

template <typename T>
Scalar
sum_safe(const Eigen::MatrixBase<T> &X)
{
    inf_zero_op<Mat> op;
    return X.unaryExpr(op).sum();
}

template <typename T1, typename T2, typename Ret>
void
XY_safe(const Eigen::MatrixBase<T1> &X,
        const Eigen::MatrixBase<T2> &Y,
        Eigen::MatrixBase<Ret> &ret)
{
    inf_zero_op<Mat> op;
    ret.derived() = X.unaryExpr(op) * Y.unaryExpr(op);
}

template <typename T1, typename T2, typename Ret>
void
XtY_safe(const Eigen::MatrixBase<T1> &X,
         const Eigen::MatrixBase<T2> &Y,
         Eigen::MatrixBase<Ret> &ret)
{
    XY_safe(X.transpose(), Y, ret);
}

#endif
