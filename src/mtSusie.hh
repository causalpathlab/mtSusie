// [[Rcpp::plugins(cpp14)]]
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <tuple>
#include <utility>
#include <unordered_map>
#include <memory>

#include "eigen_util.hh"
#include "std_util.hh"
#include "math.hh"
#include "util.hh"
#include "check.hh"
#include "tuple_util.hh"

#ifdef __cplusplus
extern "C" {
#endif

#include "bgzf.h"
#include "kstring.h"
#include "tabix.h"

#ifdef __cplusplus
}
#endif

#ifndef MT_SUSIE_HH_
#define MT_SUSIE_HH_

using Scalar = float;
using SpMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, std::ptrdiff_t>;
using Index = SpMat::Index;
using MSpMat = Eigen::MappedSparseMatrix<Scalar>;

using Mat = typename Eigen::
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using IntMat = typename Eigen::
    Matrix<std::ptrdiff_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using IntVec = typename Eigen::Matrix<std::ptrdiff_t, Eigen::Dynamic, 1>;

#include "mtSusie_util.hh"
#include "mtSusie_rcpp_util.hh"
#include "mtSusie_shared_effect.hh"
#include "mtSusie_regression.hh"
#include "mtSusie_interaction_regression.hh"

#endif
