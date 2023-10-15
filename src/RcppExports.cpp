// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// fit_mt_interaction_susie
Rcpp::List fit_mt_interaction_susie(const Rcpp::NumericMatrix x, const Rcpp::NumericMatrix y, const Rcpp::NumericMatrix w, const std::size_t levels_per_inter, const std::size_t max_iter, const std::size_t min_iter, const double coverage, const double tol, const double prior_var, Rcpp::Nullable<double> min_pip_cutoff, const bool full_stat, const bool stdize_lbf, const bool local_residual, const bool add_marginal, const bool update_prior, const bool do_hard_selection, const double hard_lodds_cutoff);
RcppExport SEXP _mtSusie_fit_mt_interaction_susie(SEXP xSEXP, SEXP ySEXP, SEXP wSEXP, SEXP levels_per_interSEXP, SEXP max_iterSEXP, SEXP min_iterSEXP, SEXP coverageSEXP, SEXP tolSEXP, SEXP prior_varSEXP, SEXP min_pip_cutoffSEXP, SEXP full_statSEXP, SEXP stdize_lbfSEXP, SEXP local_residualSEXP, SEXP add_marginalSEXP, SEXP update_priorSEXP, SEXP do_hard_selectionSEXP, SEXP hard_lodds_cutoffSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type w(wSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type levels_per_inter(levels_per_interSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type min_iter(min_iterSEXP);
    Rcpp::traits::input_parameter< const double >::type coverage(coverageSEXP);
    Rcpp::traits::input_parameter< const double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const double >::type prior_var(prior_varSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type min_pip_cutoff(min_pip_cutoffSEXP);
    Rcpp::traits::input_parameter< const bool >::type full_stat(full_statSEXP);
    Rcpp::traits::input_parameter< const bool >::type stdize_lbf(stdize_lbfSEXP);
    Rcpp::traits::input_parameter< const bool >::type local_residual(local_residualSEXP);
    Rcpp::traits::input_parameter< const bool >::type add_marginal(add_marginalSEXP);
    Rcpp::traits::input_parameter< const bool >::type update_prior(update_priorSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_hard_selection(do_hard_selectionSEXP);
    Rcpp::traits::input_parameter< const double >::type hard_lodds_cutoff(hard_lodds_cutoffSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_mt_interaction_susie(x, y, w, levels_per_inter, max_iter, min_iter, coverage, tol, prior_var, min_pip_cutoff, full_stat, stdize_lbf, local_residual, add_marginal, update_prior, do_hard_selection, hard_lodds_cutoff));
    return rcpp_result_gen;
END_RCPP
}
// fit_mt_susie
Rcpp::List fit_mt_susie(const Rcpp::NumericMatrix x, const Rcpp::NumericMatrix y, const std::size_t levels, const std::size_t max_iter, const std::size_t min_iter, const double coverage, const double tol, const double prior_var, const double lodds_cutoff, Rcpp::Nullable<double> min_pip_cutoff, const bool full_stat, const bool stdize_lbf, const bool local_residual, const bool update_prior, const bool do_hard_selection, const double hard_lodds_cutoff);
RcppExport SEXP _mtSusie_fit_mt_susie(SEXP xSEXP, SEXP ySEXP, SEXP levelsSEXP, SEXP max_iterSEXP, SEXP min_iterSEXP, SEXP coverageSEXP, SEXP tolSEXP, SEXP prior_varSEXP, SEXP lodds_cutoffSEXP, SEXP min_pip_cutoffSEXP, SEXP full_statSEXP, SEXP stdize_lbfSEXP, SEXP local_residualSEXP, SEXP update_priorSEXP, SEXP do_hard_selectionSEXP, SEXP hard_lodds_cutoffSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type y(ySEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type levels(levelsSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const std::size_t >::type min_iter(min_iterSEXP);
    Rcpp::traits::input_parameter< const double >::type coverage(coverageSEXP);
    Rcpp::traits::input_parameter< const double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const double >::type prior_var(prior_varSEXP);
    Rcpp::traits::input_parameter< const double >::type lodds_cutoff(lodds_cutoffSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type min_pip_cutoff(min_pip_cutoffSEXP);
    Rcpp::traits::input_parameter< const bool >::type full_stat(full_statSEXP);
    Rcpp::traits::input_parameter< const bool >::type stdize_lbf(stdize_lbfSEXP);
    Rcpp::traits::input_parameter< const bool >::type local_residual(local_residualSEXP);
    Rcpp::traits::input_parameter< const bool >::type update_prior(update_priorSEXP);
    Rcpp::traits::input_parameter< const bool >::type do_hard_selection(do_hard_selectionSEXP);
    Rcpp::traits::input_parameter< const double >::type hard_lodds_cutoff(hard_lodds_cutoffSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_mt_susie(x, y, levels, max_iter, min_iter, coverage, tol, prior_var, lodds_cutoff, min_pip_cutoff, full_stat, stdize_lbf, local_residual, update_prior, do_hard_selection, hard_lodds_cutoff));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mtSusie_fit_mt_interaction_susie", (DL_FUNC) &_mtSusie_fit_mt_interaction_susie, 17},
    {"_mtSusie_fit_mt_susie", (DL_FUNC) &_mtSusie_fit_mt_susie, 16},
    {NULL, NULL, 0}
};

RcppExport void R_init_mtSusie(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
