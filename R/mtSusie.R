#' @rdname mt_susie
#'
#' @title Fit multi-trait SuSiE across several regression models
#'
#' @param X n x p design matrix
#' @param Y n x m output matrix
#' @param L levels or expected num of independent factors (default: 15)
#' @param clamp handle outliers by winsorization (default: 4)
#' @param max.iter maximum iterations (default: 100)
#' @param min.iter minimum iterations (default: 5)
#' @param tol tolerance level check convergence (default: 1e-8)
#' @param coverage targeted PIP coverage per each level (default: 0.9)
#' @param min.pip.cutoff min PIP to report (default: 1/p)
#' @param lodds.cutoff log-odds ratio cutoff for multitrait factors
#' @param prior.var prior level variance
#' 
#' @export
#'
mt_susie <- function(X, Y, L=15,
                     clamp=4,
                     max.iter=100,
                     min.iter = 5,
                     tol=1e-8,
                     coverage = .9,
                     min.pip.cutoff = NULL,
                     lodds.cutoff = 0,
                     prior.var = 100) {

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
    ret <- mtSusie::fit_mt_susie(X, Y, levels=L,
                                 max_iter = max.iter,
                                 min_iter = min.iter,
                                 coverage = coverage,
                                 tol = tol,
                                 prior_var = prior.var,
                                 lodds_cutoff = lodds.cutoff,
                                 min_pip_cutoff = min.pip.cutoff)

    message("Done")
    return(ret)
}