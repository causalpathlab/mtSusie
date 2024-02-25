#' @rdname susie_cs
#'
#' @title Consolidate SuSiE credible sets across all output variables
#'
#' @param X n x p design matrix
#' @param Y n x m output matrix
#' @param L levels or expected num of independent factors (default: 5)
#' @param clamp handle outliers by winsorization (default: NULL)
#' @param max.iter maximum iterations (default: 100)
#' @param min.iter minimum iterations (default: 5)
#' @param tol tolerance level check convergence (default: 1e-8)
#' @param coverage targeted PIP coverage per each level (default: 0.9)
#' @param min.pip.cutoff min PIP to report (default: 1/p)
#' @param prior.var prior level variance (default: 1e-2)
#' @param output.full.stat output every bit of the results (default: FALSE)
#' @param local.residual calculate residual variance locally (default: FALSE)
#' @param update.prior update prior variance (default: TRUE)
#'
#' @return a list of mtSusie estimates
#' 
#' \item{alpha}{Posterior probability of variant across `p x level`}
#' \item{resid.var}{residual variance `level x traits`}
#' \item{prior.var}{prior variance `1 x traits`}
#' \item{log.odds}{log-odds ratio `level x traits`}
#' \item{mu}{a list of `p x m` mean parameters}
#' \item{var}{a list of `p x m` variance parameters}
#' \item{lbf}{a list of `p x m` log-Bayes Factors}
#' \item{z}{a list of `p x m` z-scores}
#' \item{n}{number of samples}
#' \item{m}{number of traits/outputs}
#' \item{p}{number of variants}
#' \item{L}{number of layers/levels}
#' \item{loglik}{log-likelihood trace}
#' \item{cs}{credible sets (use `data.table::setDT` to assemble)}
#'
#' In the `cs`, we have
#' \item{variants}{variant indexes `[1 .. p]`}
#' \item{traits}{trait indexes `[1 .. m]`}
#' \item{levels}{levels `[1 .. L]`}
#' \item{alpha}{shared PIP}
#' \item{mean}{univariate mean}
#' \item{var}{univariate var}
#' \item{z}{univariate z-score}
#' \item{lfsr}{local false sign rate}
#' \item{lodds}{log-odds ratio}
#' \item{interaction}{interacting column index (if W!=NULL)}
#'
#' @export
#' 
susie_cs <- function(X, Y, L = 5,
                     clamp = NULL,
                     max.iter = 100,
                     min.iter = 5,
                     tol = 1e-8,
                     coverage = .9,
                     min.pip.cutoff = NULL,
                     prior.var = 1e-2,
                     output.full.stat = FALSE,
                     local.residual = FALSE,
                     update.prior = TRUE){

    message("Fitting Susie ...")

    ## first susie model
    .temp <- fit_mt_susie(X, Y[, 1, drop = F], levels=L,
                          max_iter = max.iter,
                          min_iter = min.iter,
                          coverage = coverage,
                          tol = tol,
                          prior_var = prior.var,
                          min_pip_cutoff = min.pip.cutoff,
                          full_stat = output.full.stat,
                          local_residual = local.residual,
                          update_prior = update.prior)

    ret <- .temp$cs

    if(ncol(Y) > 1){
        for(jj in 2:ncol(Y)){

            .temp <- fit_mt_susie(X, Y[, jj, drop = F], levels=L,
                                  max_iter = max.iter,
                                  min_iter = min.iter,
                                  coverage = coverage,
                                  tol = tol,
                                  prior_var = prior.var,
                                  min_pip_cutoff = min.pip.cutoff,
                                  full_stat = output.full.stat,
                                  local_residual = local.residual,
                                  update_prior = update.prior)

            ret.j <- .temp$cs
            nn.j <- max(sapply(ret.j, length))
            ret.j[["traits"]] <- rep(jj, nn.j)

            for(k in names(ret)){
                ret[[k]] <- c(ret[[k]], ret.j[[k]])
            }
        }
    }
    
    message("Done")
    return(ret)
}

#' @rdname mt_susie
#'
#' @title Fit multi-trait SuSiE across several regression models
#'
#' @param X n x p design matrix
#' @param Y n x m output matrix
#' @param L levels or expected num of independent factors (default: 5)
#' 
#' @param max.iter maximum iterations (default: 100)
#' @param min.iter minimum iterations (default: 5)
#' @param tol tolerance level check convergence (default: 1e-8)
#' @param coverage targeted PIP coverage per each level (default: 0.9)
#' @param min.pip.cutoff min PIP to report (default: 1/p)
#' @param prior.var prior level variance (default: 1e-2)
#' @param update.prior update prior variance (default: TRUE)
#' @param output.full.stat output every bit of the results (default: FALSE)
#' @param local.residual calculate residual variance locally (default: FALSE)
#'
#' @return a list of mtSusie estimates
#' 
#' \item{alpha}{Posterior probability of variant across `p x level`}
#' \item{resid.var}{residual variance `level x traits`}
#' \item{prior.var}{prior variance `1 x traits`}
#' \item{log.odds}{log-odds ratio `level x traits`}
#' \item{mu}{a list of `p x m` mean parameters}
#' \item{var}{a list of `p x m` variance parameters}
#' \item{lbf}{a list of `p x m` log-Bayes Factors}
#' \item{z}{a list of `p x m` z-scores}
#' \item{n}{number of samples}
#' \item{m}{number of traits/outputs}
#' \item{p}{number of variants}
#' \item{L}{number of layers/levels}
#' \item{loglik}{log-likelihood trace}
#' \item{cs}{credible sets (use `data.table::setDT` to assemble)}
#'
#' In the `cs`, we have
#' \item{variant}{variant indexes `[1 .. p]`}
#' \item{trait}{trait indexes `[1 .. m]`}
#' \item{level}{levels `[1 .. L]`}
#' \item{alpha}{shared PIP}
#' \item{mean}{univariate mean}
#' \item{var}{univariate var}
#' \item{z}{univariate z-score}
#' \item{lfsr}{local false sign rate}
#' \item{lodds}{log-odds ratio}
#' 
#' @import Rcpp
#' @import RcppEigen
#' 
#' @export
#'
mt_susie <- function(X, Y, L=5,
                     max.iter = 100,
                     min.iter = 5,
                     tol = 1e-8,
                     coverage = .9,
                     min.pip.cutoff = NULL,
                     prior.var = 1e-2,
                     output.full.stat = FALSE,
                     local.residual = FALSE,
                     update.prior = TRUE) {

    message("Fitting mtSusie ...")

    ret <- fit_mt_susie(X, Y, levels=L,
                        max_iter = max.iter,
                        min_iter = min.iter,
                        coverage = coverage,
                        tol = tol,
                        prior_var = prior.var,
                        min_pip_cutoff = min.pip.cutoff,
                        full_stat = output.full.stat,
                        local_residual = local.residual,
                        update_prior = update.prior)

    message("Done")
    return(ret)
}

#' @rdname mt_susie_inter
#'
#' @title Fit multi-trait SuSiE with interaction effects
#'
#' @param X n x p design matrix
#' @param Y n x m output matrix
#' @param W n x k interaction terms matrix
#' @param L.wx levels for each interaction W*X (default: 5)
#' @param L.x levels for X (default: 1)
#' @param L.w levels for W (default: 1)
#' 
#' @param max.iter maximum iterations (default: 100)
#' @param min.iter minimum iterations (default: 5)
#' @param tol tolerance level check convergence (default: 1e-8)
#' @param coverage targeted PIP coverage per each level (default: 0.9)
#' @param min.pip.cutoff min PIP to report (default: 1/p)
#' 
#' @param prior.var prior level variance (default: 1e-2)
#' @param update.prior update prior variance (default: TRUE)
#' @param output.full.stat output every bit of the results (default: FALSE)
#' @param local.residual calculate residual variance locally (default: FALSE)
#'
#' @return a list of mtSusie estimates
#' 
#' \item{alpha}{Posterior probability of variant across `p x level`}
#' \item{resid.var}{residual variance `level x traits`}
#' \item{prior.var}{prior variance `1 x traits`}
#' \item{log.odds}{log-odds ratio `level x traits`}
#' \item{mu}{a list of `p x m` mean parameters}
#' \item{var}{a list of `p x m` variance parameters}
#' \item{lbf}{a list of `p x m` log-Bayes Factors}
#' \item{z}{a list of `p x m` z-scores}
#' \item{n}{number of samples}
#' \item{m}{number of traits/outputs}
#' \item{p}{number of variants}
#' \item{L}{number of layers/levels}
#' \item{loglik}{log-likelihood trace}
#' \item{cs}{credible sets (use `data.table::setDT` to assemble)}
#'
#' In the `cs`, we have
#' \item{variant}{variant indexes `[1 .. p]`}
#' \item{trait}{trait indexes `[1 .. m]`}
#' \item{level}{levels `[1 .. L]`}
#' \item{alpha}{shared PIP}
#' \item{mean}{univariate mean}
#' \item{var}{univariate var}
#' \item{z}{univariate z-score}
#' \item{lfsr}{local false sign rate}
#' \item{lodds}{log-odds ratio}
#' \item{interaction}{interacting column index}
#' 
#' @import Rcpp
#' @import RcppEigen
#' 
#' @export
#'
mt_susie_inter <- function(X, Y, W,
                           L.wx = 5,
                           L.x = 1,
                           L.w = 1,
                           clamp = NULL,
                           max.iter = 100,
                           min.iter = 5,
                           tol = 1e-8,
                           coverage = .9,
                           min.pip.cutoff = NULL,
                           prior.var = 1e-2,
                           output.full.stat = FALSE,
                           local.residual = FALSE,
                           update.prior = TRUE) {

    message("Fitting mtSusie ...")

    ret <- fit_mt_interaction_susie(X, Y, W,
                                    levels_per_wx = L.wx,
                                    levels_per_w = L.w,
                                    levels_per_x = L.x,
                                    max_iter = max.iter,
                                    min_iter = min.iter,
                                    coverage = coverage,
                                    tol = tol,
                                    prior_var = prior.var,
                                    min_pip_cutoff = min.pip.cutoff,
                                    full_stat = output.full.stat,
                                    local_residual = local.residual)

    message("Done")
    return(ret)
}
