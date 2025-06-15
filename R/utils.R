#' Compute marginal log-likelihood (via Importance Sampling) for an object of class "pmCDM".
#'
#' @param mod \code{pmCDM} object.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{llk}: Marginal log-likelihood (computed via Monte-Carlo integration) evaluated at \code{A}, \code{C}, and \code{R}.
#' }
#' @details Test
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
logLik.pmCDM <- function(mod, Y, control = list(), ...){
  if("gapmCDM" %in% class(mod)){
    if(!is.null(mod$llk)){
      return(mod$llk)
    } else {
      control = pr_control_gaCDM(control,...)
      control$basis = match.arg(control$basis, c("is","bs","pwl"))
      if(control$basis == "pwl" & is.null(control$degree)) control$degree = 1
      if(control$basis != "pwl" & is.null(control$degree)) control$degree = 2
      if(control$verbose) cat(" Model: Generalized Additive PM-CDM")
      llk = fy_gapmCDM_IS(Y = Y[],A = mod$A[],C = mod$C[],mu = mod$mu[],R = mod$R[],
                          pM = mod$posMu[], pR = mod$posR[], control = control)
      return(llk)
    }
  }
  else if("apmCDM" %in% class(mod)){
    if(!is.null(mod$llk)){
      return(mod$llk)
    } else {
      control = pr_control_aCDM(control,...)
      if(control$verbose) cat(" Model: Additive PM-CDM")
      q = ncol(mod$R)
      Apat = as.matrix(expand.grid(lapply(1:q,function(x) c(0,1))))
      llk = fy_aCDM_IS(Y = Y[],G = mod$G[], Qmatrix = mod$Qmatrix, Apat = Apat[],
                       mu = mod$mu[],R = mod$R[],
                       pM = mod$posMu[], pR = mod$posR[], control = control)
      return(llk)
    }
  }
}
