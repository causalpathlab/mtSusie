#' @rdname mt_susie
#'
#' @title Fit multi-trait SuSiE across several regression models
#'
#' @param X n x p design matrix
#' @param Y n x m output matrix
#' @param W n x k interaction terms matrix (default: NULL)
#' @param L levels or expected num of independent factors (default: 5)
#' @param clamp handle outliers by winsorization (default: 4)
#' @param max.iter maximum iterations (default: 100)
#' @param min.iter minimum iterations (default: 5)
#' @param tol tolerance level check convergence (default: 1e-8)
#' @param coverage targeted PIP coverage per each level (default: 0.9)
#' @param min.pip.cutoff min PIP to report (default: 1/p)
#' @param lodds.cutoff log-odds ratio cutoff for multi-trait factors (default: 0)
#' @param prior.var prior level variance (default: 0.1)
#' @param output.full.stat output every bit of the results (default: TRUE)
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
#' 
#' \item{interaction}{interacting column index} (if W!=NULL) 
#' 
#' 
#' @export
#'
mt_susie <- function(X, Y, L=5,
                     W = NULL,
                     clamp = 4,
                     max.iter = 100,
                     min.iter = 5,
                     tol = 1e-8,
                     coverage = .9,
                     min.pip.cutoff = NULL,
                     lodds.cutoff = 0,
                     prior.var = .1,
                     output.full.stat = TRUE) {

    xx <- apply(X, 2, scale)
    yy <- apply(Y, 2, scale)

    if(!is.null(clamp)){
        yy[yy > clamp] <- clamp
        yy[yy < -clamp] <- -clamp
        yy <- apply(yy, 2, scale)

        xx[xx > clamp] <- clamp
        xx[xx < -clamp] <- -clamp
        xx <- apply(xx, 2, scale)

        message("winsorization with the clamping value = ", clamp)
    }

    message("Fitting mtSusie ...")

    if(is.null(W)){
        ret <- fit_mt_susie(X, Y, levels=L,
                            max_iter = max.iter,
                            min_iter = min.iter,
                            coverage = coverage,
                            tol = tol,
                            prior_var = prior.var,
                            lodds_cutoff = lodds.cutoff,
                            min_pip_cutoff = min.pip.cutoff,
                            full_stat = output.full.stat)
    } else {
        lvl <- max(L, ceiling(L / ncol(W)) + 1)
        ret <- fit_mt_interaction_susie(X, Y, W,
                                        levels_per_inter = lvl,
                                        max_iter = max.iter,
                                        min_iter = min.iter,
                                        coverage = coverage,
                                        tol = tol,
                                        prior_var = prior.var,
                                        lodds_cutoff = lodds.cutoff,
                                        min_pip_cutoff = min.pip.cutoff,
                                        full_stat = output.full.stat)
    }

    message("Done")
    return(ret)
}
