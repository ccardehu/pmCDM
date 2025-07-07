#' Fit a Generalized Additive Partial-Mastery Cognitive Diagnosis Model (GaPM-CDM).
#'
#' @param data Matrix \code{(n x p)} with binary entries. Missing data should be coded as \code{NA}.
#' @param q Number of latent variables.
#' @param control List of control parameters (see 'Details').
#' @param start.par (optional) For simulation use. List of size 4 with starting model parameters (A, C, D, R).
#' @param ... Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{A}: A matrix \code{(p x q)} of estimated contribution ('weights') parameters.
#'  \item \code{C}: An array \code{(p x tp x q)} of estimated I/B-spline coefficients \code{(tp = degree + |knots|)}.
#'  \item \code{R}: A matrix \code{(q x q)} of estimated correlations for the latent variables
#'  \item \code{llk}: Marginal log-likelihood (computed via Monte-Carlo integration) evaluated at \code{A}, \code{C}, and \code{R}.
#'  \item \code{AIC}: Akaike information criterion for the estimated model.
#'  \item \code{BIC}: Bayesian (Schwarz) information criterion for the estimated model.
#'  \item \code{cdllk.trace}: (if return.trace = T) A matrix \code{(iter x 2)} with the trace for the observed variables and latent variables log-likelihood.
#'  \item \code{ar.trace}: (if return.trace = T) A vector \code{(iter)} with the trace for the acceptance rate (for MALA and RWMH samplers).
#'  \item \code{theta.trace}: (if return.trace = T) A matrix \code{(iter x dim(theta))} with the trace for the parameter estimates.
#' }
#' @details Test
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
gapmCDM <- function(data,q,control = list(), start.par = NULL, ...){

  control = pr_control_gaCDM(control,...)
  if(control$verbose) cat(" Model: Generalized Additive PM-CDM \n")
  control$sampler = match.arg(control$sampler, c("ULA","MALA","RWMH"))
  control$basis = match.arg(control$basis, c("is","bs","pwl"))
  control$algorithm = match.arg(control$algorithm, c("GD","ADAM","mixed"))
  if(is.null(control$degree) && control$basis == "pwl") control$degree = 1
  if(is.null(control$degree) && control$basis != "pwl") control$degree = 2
  if(is.null(control$Qmatrix)) control$Qmatrix = matrix(1,p,q)
  p = ncol(data)
  tp = length(control$knots) + control$degree

  if(is.list(start.par)){
    if(length(start.par) != 5) stop("Argument `start.par' needs to be a list with elements `A', `C', `D', `mu', and `R'.")
    pp = start.par
    if(nrow(pp$A) != ncol(data)) stop("Matrix `start.par$A' mismatch rows with p.")
    if(ncol(pp$C) != ncol(pp$D)) stop("Mismatch columns in `start.par$C' and `start.par$D'.")
    if(ncol(pp$C) != tp) stop("Matrix `start.par$C' mismatch rows with length(control$knots) + control$degree (+1 if intercept)")
    if(length(pp$mu) != nrow(pp$R)) stop("Lenght of `start.par$mu' and nrows of `start.par$R' differ.")
  }
  if(!is.null(start.par) && !is.list(start.par) && (start.par == "random")){
    if(control$verbose) cat(" Generating random starting values for model parameters ...")
    control$prob.sparse = 0.0
    control$iden.R = T
    if(!is.null(control$seed)) set.seed(control$seed)
    pp = pr_param_gaCDM(p,q,tp,T,control)
    if(control$verbose) cat("\r Generating random starting values for model parameters ... (Done!) \n")
  }
  if(is.null(start.par)){
    pp = pr_param_gaCDM(p,q,tp,F,control)
  }

  if(!is.null(control$start.zn) && is.matrix(control$start.zn)){
    zn = control$start.zn
  } else if(control$start.zn == "random") {
    if(control$verbose) cat(" Generating random starting values for latent variables ...")
    zn = mvtnorm::rmvnorm(nrow(data),mean = rep(0,q))
    if(control$verbose) cat("\r Generating random starting values for latent variables ... (Done!) \n")
  } else if(control$start.zn == "fa") {
    if(control$verbose) cat(" Generating starting values for latent variables via Factor Analysis ...")
    tmp = suppressWarnings(psych::fa(r = data, nfactors = q, cor = "tet", fm = "ml", rotate = "oblimin",
                                     missing = T))
    zn = tmp$scores
    if(control$cor.R) pp$R = cor(zn) else pp$R = cov(zn)
    if(control$verbose) cat("\r Generating starting values for latent variables via Factor Analysis ... (Done!) \n")
  }

  if(!is.null(control$seed)) set.seed(control$seed)
  out <- gapmCDM_fit_rcpp(Y = data[],A = pp$A[],C = pp$C[],D = pp$D[],mu = pp$mu[],R = pp$R[], Z = zn[], control = control)
  colnames(out$PI) <- rownames(out$A) <- rownames(out$C) <- colnames(data)
  colnames(out$Z) <- colnames(out$A) <- colnames(out$R) <- rownames(out$R) <- names(out$mu) <- paste0("Z",1:q)
  colnames(out$U) <- paste0("U",1:q)
  if(control$return.trace){
    colnames(out$cdllk.trace) <- c("fz","fyz")
    Anames <- paste0("A",apply(expand.grid(paste0("j",1:p),paste0("k",1:q)),1,paste,collapse = "."))
    Cnames <- paste0("C",apply(expand.grid(apply(expand.grid(paste0("j",1:p),paste0("r",1:ncol(out$C))),1,paste,collapse = "."), paste0("k",1:q)),1,paste,collapse = "."))
    Mnames <- paste0("mu",1:q)
    Rnames <- paste0("R",apply(which(lower.tri(diag(q)) == T,arr.ind = T),1,paste0,collapse = ""))
    colnames(out$theta.trace) <- c(Anames,Cnames,Mnames,Rnames)
  }
  class(out) = c("gapmCDM", "pmCDM")
  return(out)
}

#' Simulate data from a Generalized Additive Partial-Mastery Cognitive Diagnosis Model (GaPM-CDM).
#'
#' @param n Number of simulated entries.
#' @param p Number of observed (binary) variables.
#' @param q Number of latent variables.
#' @param control List of control parameters (see 'Details').
#' @param start.par (optional) List of size 4 with starting model parameters (A, C, D, R).
#' @param ... Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{Y}: A matrix \code{(n x p)} of simulated observed (binary) variables.
#'  \item \code{Z}: A matrix \code{(n x q)} of simulated latent variables (on the continuous scale).
#'  \item \code{U}: A matrix \code{(n x q)} of simulated latent variables (on the [0,1] scale).
#'  \item \code{spM}: A matrix \code{(n x tp)} of I/B-spline basis functions (\code{(tp = degree + |knots|)}).
#'  \item \code{PI}: A matrix \code{(n x p)} of predicted probabilities.
#' }
#' @details Test
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
gapmCDM_sim <- function(n, p, q, control = list(),
                        start.par = NULL, ...){
  control = pr_controlsim_gaCDM(control,q,...)
  control$basis = match.arg(control$basis, c("is","bs","pwl"))
  if(is.null(control$degree) && control$basis == "pwl") control$degree = 1
  if(is.null(control$degree) && control$basis != "pwl") control$degree = 2
  if(is.null(control$Qmatrix)) control$Qmatrix = matrix(1,p,q)
  tp = length(control$knots) + control$degree
  if(!is.null(start.par) && !is.list(start.par) && (length(start.par) != 5))
    stop("Argument `start.par' needs to be a list with elements `A', `C', `D', `mu', and `R'.")
  if(!is.null(start.par)){
    pp = start.par
    if(nrow(pp$A) != p) stop("Matrix `start.par$A' mismatch rows with p.")
    if(ncol(pp$C) != ncol(pp$D)) stop("Mismatch columns in `start.par$C' and `start.par$D'.")
    if(ncol(pp$C) != tp) stop("Matrix `start.par$C' mismatch rows with length(control$knots) + control$degree (+1 if intercept)")
    if(length(pp$mu) != nrow(pp$R)) stop("Lenght of `start.par$mu' and nrows of `start.par$R' differ.")
    if(rowSums(pp$A * control$Qmatrix) != p) stop("Check configuration of `start.par$A' and `control$Qmatrix'.")
  } else {
    pp = pr_param_gaCDM(p,q,tp,T,control)
  }
  if(!is.null(control$seed)) set.seed(control$seed)
  out <- gapmCDM_sim_rcpp(n,q,p,pp$A,pp$C,pp$mu,pp$R,control)
  out$A <- pp$A
  out$C <- pp$C
  out$D <- pp$D
  out$mu <- pp$mu
  out$R <- pp$R
  out$posMu <- matrix(colMeans(out$Z),nrow = n, ncol = q, byrow = T)
  out$posR <- array(cov(out$Z),dim = c(q,q,n))
  rownames(out$A) <- rownames(out$C) <- rownames(out$D) <- colnames(out$Y) <- colnames(out$PI) <- paste0("Y",1:p)
  colnames(out$A) <- colnames(out$R) <- rownames(out$R) <- colnames(out$posR) <- rownames(out$posR) <- colnames(out$Z) <- names(out$mu) <- names(out$posMu) <- paste0("Z",1:q)
  colnames(out$U) <- paste0("U",1:q)
  class(out) = c("gapmCDM", "pmCDM")
  return(out)
}

#' Find number of latent variables (q) using Universal inference
#'
#' @param data Matrix \code{(n x p)} with binary entries. Missing data should be coded as \code{NA}.
#' @param qmax Maximum number of latent variables to test.
#' @param qmin Minimum number of latent variables to test.
#' @param alpha Critical value for the universal inference test.
#' @param control List of control parameters (see 'Details').
#' @param type Type of test for Universal inference.
#' @param ... Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{q}: Number of latent variables when Ho: dim(Z) = q is not rejected.
#' }
#' @details Test
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
gapmCDM_findqUI <- function(data, qmax = 10, qmin = 2, alpha = 0.05, type = "cross-fit",
                            control = list(), ...){
  type = match.arg(type, c("split", "cross-fit"))
  control = pr_control_gaCDM(control,...)
  controlA = control
  controlA$verbose = F
  control$nsim = 0
  p = ncol(data)
  tp = length(control$knots) + control$degree

  if(!is.null(control$seed)) set.seed(control$seed)

  if(type == "split"){
    for(q in qmin:qmax){
      if(control$verbose) cat(paste0("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
                                     "\n Sequential testing (type ",type,"):",
                                     "\n Ho: q = ",q," vs. Ha: q = ",q+1,
                                     "\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"))
      D0 = sample(c(T,F), ceiling(nrow(data)), replace = T)
      D1 = !D0
      if(control$verbose) cat(paste0("\n Generating random starting values for latent variables q = ", q+1,", split D1 ..."))
      znD1 = mvtnorm::rmvnorm(nrow(data[D1,]),mean = rep(0,q+1))
      if(control$verbose) cat(paste0("\r Generating random starting values for latent variables q = ", q+1,", split D1 ... (Done!)"))
      ppD1 = pr_param_gaCDM(p,q+1,tp,F,control)
      outD1 <- gapmCDM_fit_rcpp(Y = data[D1,],A = ppD1$A[],C = ppD1$C[],D = ppD1$D[],mu = ppD1$mu[],R = ppD1$R[], Z = znD1[], control = control)

      if(control$verbose) cat(paste0("\n Generating random starting values for latent variables q = ", q,", split D0 ..."))
      znD0 = mvtnorm::rmvnorm(nrow(data[D0,]),mean = rep(0,q))
      if(control$verbose) cat(paste0("\r Generating random starting values for latent variables q = ", q,", split D0 ... (Done!)"))
      ppD0 = pr_param_gaCDM(p,q,tp,F,control)
      outD0 <- gapmCDM_fit_rcpp(Y = data[D0,],A = ppD0$A[],C = ppD0$C[],D = ppD0$D[],mu = ppD0$mu[],R = ppD0$R[], Z = znD0[], control = control)

      if(control$verbose) cat(paste0("\n Computing marginal log-likelihood on D0 for comparison (q = ",q," vs. q = ",q+1,") ..."))
      L0 = fy_gapmCDM(data[D0,], outD0$A[], outD0$C[], outD0$R[], controlA)
      L1 = fy_gapmCDM(data[D0,], outD1$A[], outD1$C[], outD1$R[], controlA)
      if(control$verbose) cat(paste0("\r Computing marginal log-likelihood on D0 for comparison (q = ",q," vs. q = ",q+1,") ... (Done!)"))
      tt = L1 - L0
      if(control$verbose) cat(paste0("\n log(L1): ", round(L1,4), ", log(L0): ", round(L0,4),
                                     ". tt = log(L1/L0): ", round(tt,4) ," vs. log(1/alpha): ", round(log(1/alpha),4), "\n"))
      if(tt <= log(1/alpha)){
        if(control$verbose) cat(paste0("\n Ho: q = ",q," not rejected. Selected q = ",q,"\n"))
        break
      } else {
        if(control$verbose) cat(paste0("\n Ho: q = ",q," rejected. Testing for Ho: q = ",q+1," ... \n"))
        if(q == qmax & control$verbose) cat(paste0("\n `qmax' reached. Increase qmax and test for q > ",q+1))
      }
    }
  } else if(type == "cross-fit"){
    for(q in qmin:qmax){
      if(control$verbose) cat(paste0("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
                                     "\n Sequential testing (type ",type,"):",
                                     "\n Ho: q = ",q," vs. Ha: q = ",q+1,
                                     "\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"))
      D0 = sample(c(T,F), ceiling(nrow(data)), replace = T)
      D1 = !D0

      if(control$verbose) cat(paste0("\n Generating random starting values for latent variables q = ", q+1,", split D1 ..."))
      znD1a = mvtnorm::rmvnorm(nrow(data[D1,]),mean = rep(0,q+1))
      if(control$verbose) cat(paste0("\r Generating random starting values for latent variables q = ", q+1,", split D1 ... (Done!)"))
      if(control$verbose) cat(paste0("\n Generating random starting values for latent variables q = ", q+1,", split D0 ..."))
      znD0a = mvtnorm::rmvnorm(nrow(data[D0,]),mean = rep(0,q+1))
      if(control$verbose) cat(paste0("\r Generating random starting values for latent variables q = ", q+1,", split D0 ... (Done!)"))

      ppq1 = pr_param_gaCDM(p,q+1,tp,F,control)
      if(control$verbose) cat(paste0("\n Estimating model with q = ", q+1," for split D1 ... \n"))
      if(!is.null(control$seed)) set.seed(control$seed)
      outD1a <- gapmCDM_fit_rcpp(Y = data[D1,],A = ppq1$A[],C = ppq1$C[],D = ppq1$D[],mu = ppq1$mu[],R = ppq1$R[], Z = znD1a[], control = control)
      if(control$verbose) cat(paste0("\n Estimating model with q = ", q+1," for split D0 ... \n"))
      if(!is.null(control$seed)) set.seed(control$seed)
      outD0a <- gapmCDM_fit_rcpp(Y = data[D0,],A = ppq1$A[],C = ppq1$C[],D = ppq1$D[],mu = ppq1$mu[],R = ppq1$R[], Z = znD0a[], control = control)

      if(control$verbose) cat(paste0("\n Generating random starting values for latent variables q = ", q,", split D0 ..."))
      znD0b = mvtnorm::rmvnorm(nrow(data[D0,]),mean = rep(0,q))
      if(control$verbose) cat(paste0("\r Generating random starting values for latent variables q = ", q,", split D0 ... (Done!)"))
      if(control$verbose) cat(paste0("\n Generating random starting values for latent variables q = ", q,", split D1 ..."))
      znD1b = mvtnorm::rmvnorm(nrow(data[D1,]),mean = rep(0,q))
      if(control$verbose) cat(paste0("\r Generating random starting values for latent variables q = ", q,", split D1 ... (Done!)"))

      ppq0 = pr_param_gaCDM(p,q,tp,F,control)
      if(control$verbose) cat(paste0("\n Estimating model with q = ", q," for split D0 ... \n"))
      if(!is.null(control$seed)) set.seed(control$seed)
      outD0b <- gapmCDM_fit_rcpp(Y = data[D0,],A = ppq0$A[],C = ppq0$C[],D = ppq0$D[],mu = ppq0$mu[],R = ppq0$R[], Z = znD0b[], control = control)
      if(control$verbose) cat(paste0("\n Estimating model with q = ", q," for split D1 ... \n"))
      if(!is.null(control$seed)) set.seed(control$seed)
      outD1b <- gapmCDM_fit_rcpp(Y = data[D1,],A = ppq0$A[],C = ppq0$C[],D = ppq0$D[],mu = ppq0$mu[],R = ppq0$R[], Z = znD1b[], control = control)

      if(control$verbose) cat(paste0("\n Computing marginal log-likelihoods on D0 and D1 for comparison (q = ",q," vs. q = ",q+1,") ..."))
      L0a = fy_gapmCDM(data[D0,], outD0b$A[], outD0b$C[], outD0b$mu[], outD0b$R[], controlA)
      L1a = fy_gapmCDM(data[D0,], outD1a$A[], outD1a$C[], outD1a$mu[], outD1a$R[], controlA)
      L0b = fy_gapmCDM(data[D1,], outD1b$A[], outD1b$C[], outD1b$mu[], outD1b$R[], controlA)
      L1b = fy_gapmCDM(data[D1,], outD0a$A[], outD0a$C[], outD0a$mu[], outD0a$R[], controlA)
      tt = log((exp(L1a - L0a) + exp(L1b - L0b))/2)
      if(control$verbose) cat(paste0("\r Computing marginal log-likelihoods on D0 and D1 for comparison (q = ",q," vs. q = ",q+1,") ... (Done!) \n"))
      if(control$verbose) cat(paste0("\n log(L1a): ", round(L1a,4), ", log(L0a): ", round(L0a,4),
                                     "\n log(L1b): ", round(L1b,4), ", log(L0b): ", round(L0b,4),
                                     "\n tt = log((L1a/L0a + L1b/Lb0)/2): ", round(tt,4) ," vs. log(1/alpha): ", round(log(1/alpha),4), "\n"))
      if(tt <= log(1/alpha)){
        if(control$verbose) cat(paste0("\n Ho: q = ",q," not rejected. Selected q = ",q,"\n"))
        break
      } else {
        if(control$verbose) cat(paste0("\n Ho: q = ",q," rejected. Testing for Ho: q = ",q+1," ... \n"))
        if(q == qmax & control$verbose) cat(paste0("\n `qmax' reached. Increase qmax and test for q > ",q+1))
      }
    }
  }
  return(q)
}

#' Find number of latent variables (q) using cross-validation (GaPM-CDM)
#'
#' @param Ytrain Matrix with observed binary entries to train the model. Missing entries (\code{Ytest}) coded as \code{NA}.
#' @param Ytest Matrix with observed binary entries to test the model. All entries are \code{NA} but the missing in \code{Ytrain}.
#' @param q Number of latent variables.
#' @param control List of control parameters (see 'Details').
#' @param ... Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{CV.error}: Cross-validation error.
#'  \item \code{AUC}: Area under the curve error.
#'  \item \code{llk.gapmCDM} Marginal log-likelihood for the gapmCDM model
#' }
#' @details Define CV.error.
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
gapmCDM_findqCV <- function(Ytrain, Ytest, q, control = list(), start.par = NULL, ...){

  control = pr_control_gaCDM(control,...)
  if(control$verbose) cat(" Model: Generalized Additive PM-CDM \n")
  control$sampler = match.arg(control$sampler, c("ULA","MALA","RWMH"))
  control$basis = match.arg(control$basis, c("is","bs","pwl"))
  control$algorithm = match.arg(control$algorithm, c("GD","ADAM","mixed"))
  if(is.null(control$degree) && control$basis == "pwl") control$degree = 1
  if(is.null(control$degree) && control$basis != "pwl") control$degree = 2
  if(is.null(control$Qmatrix)) control$Qmatrix = matrix(1,p,q)
  p = ncol(data)
  tp = length(control$knots) + control$degree

  if(is.list(start.par)){
    if(length(start.par) != 5) stop("Argument `start.par' needs to be a list with elements `A', `C', `D', `mu', and `R'.")
    ppD1 = start.par
    if(nrow(ppD1$A) != ncol(data)) stop("Matrix `start.par$A' mismatch rows with p.")
    if(ncol(ppD1) != ncol(ppD1$D)) stop("Mismatch columns in `start.par$C' and `start.par$D'.")
    if(ncol(ppD1$C) != tp) stop("Matrix `start.par$C' mismatch rows with length(control$knots) + control$degree (+1 if intercept)")
    if(length(ppD1$mu) != nrow(ppD1$R)) stop("Lenght of `start.par$mu' and nrows of `start.par$R' differ.")
  }
  if(!is.null(start.par) && !is.list(start.par) && (start.par == "random")){
    if(control$verbose) cat(" Generating random starting values for model parameters ...")
    control$prob.sparse = 0.0
    control$iden.R = T
    if(!is.null(control$seed)) set.seed(control$seed)
    ppD1 = pr_param_gaCDM(p,q,tp,T,control)
    if(control$verbose) cat("\r Generating random starting values for model parameters ... (Done!) \n")
  }
  if(is.null(start.par)){
    ppD1 = pr_param_gaCDM(p,q,tp,F,control)
  }

  if(!is.null(control$start.zn) && is.matrix(control$start.zn)){
    zn = control$start.zn
  } else if(control$start.zn == "random") {
    if(control$verbose) cat(" Generating random starting values for latent variables ...")
    zn = mvtnorm::rmvnorm(nrow(Ytrain),mean = rep(0,q))
    if(control$verbose) cat("\r Generating random starting values for latent variables ... (Done!) \n")
  } else if(control$start.zn == "fa") {
    if(control$verbose) cat(" Generating starting values for latent variables via Factor Analysis ...")
    tmp = suppressWarnings(psych::fa(r = Ytrain, nfactors = q, cor = "tet", fm = "ml", rotate = "oblimin",
                                     missing = T))
    zn = tmp$scores
    if(control$cor.R) ppD1$R = cor(zn) else ppD1$R = cov(zn)
    if(control$verbose) cat("\r Generating starting values for latent variables via Factor Analysis ... (Done!) \n")
  }

  if(!is.null(control$seed)) set.seed(control$seed)
  if(control$verbose) cat(paste0("\n [Training data] \n"))
  fit1 = gapmCDM_fit_rcpp(Y = Ytrain[],A = ppD1$A[],C = ppD1$C[],D = ppD1$D[],mu = ppD1$mu[], R = ppD1$R[], Z = zn[], control = control)
  out1 = gapmCDM_cv_rcpp(Ytrain[], Ytest[],A = fit1$A[],C = fit1$C[],mu = fit1$mu[],R = fit1$R[], Z = fit1$Z[], control = control)
  pred = ROCR::prediction(out1$Yhat,out1$Yobs)
  aucCV = unlist(methods::slot(ROCR::performance(pred, "auc"), "y.values"))

  return(list(CV.error = out1$CV.error, AUC = aucCV, llk.train = fit1$llk,
              mod.train = fit1))
}


#' Find number of latent variables (q) using cross-validated MLLK
#'
#' @param Ytrain Matrix with observed binary entries to train the model. Missing entries (\code{Ytest}) coded as \code{NA}.
#' @param Ytest Matrix with observed binary entries to test the model. All entries are \code{NA} but the missing in \code{Ytrain}.
#' @param q Number of latent variables.
#' @param control List of control parameters (see 'Details').
#' @param ... Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{mllk.test}: Test data marginal log-likelihood.
#'  \item \code{mllk.train}: Train data marginal log-likelihood.
#'  \item \code{mllk.BIC} Train data BIC.
#' }
#' @details Define CV.error.
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
gapmCDM_mllkCV <- function(Ytrain, Ytest, q, control = list(), ...){
  control = pr_control_gaCDM(control,...)
  if(control$verbose) cat(" Model: Generalized Additive PM-CDM \n")
  if(control$verbose) cat(paste0("\n [ Training data ] \n\n"))
  control$sampler = match.arg(control$sampler, c("ULA","MALA","RWMH"))
  control$basis = match.arg(control$basis, c("is","bs","pwl"))
  if(is.null(control$degree) && control$basis == "pwl") control$degree = 1
  if(is.null(control$degree) && control$basis != "pwl") control$degree = 2
  if(is.null(control$Qmatrix)) control$Qmatrix = matrix(1,p,q)

  p = ncol(Ytrain)
  tp = length(control$knots) + control$degree
  ppD1 = pr_param_gaCDM(p,q,tp,F,control)

  if(!is.null(control$start.zn) && is.matrix(control$start.zn)){
    zn = control$start.zn
  } else if(control$start.zn == "random") {
    if(control$verbose) cat(" Generating random starting values for latent variables ...")
    zn = mvtnorm::rmvnorm(nrow(Ytrain),mean = rep(0,q))
    if(control$verbose) cat("\r Generating random starting values for latent variables ... (Done!) \n")
  } else if(control$start.zn == "fa") {
    if(control$verbose) cat(" Generating starting values for latent variables via Factor Analysis ...")
    tmp = suppressMessages(suppressWarnings(psych::fa(r = Ytrain, nfactors = q, cor = "tet", fm = "ml", rotate = "oblimin",
                                                      missing = T)))
    zn = tmp$scores
    if(control$cor.R) ppD1$R = cor(zn) else ppD1$R = cov(zn)
    if(control$verbose) cat("\r Generating starting values for latent variables via Factor Analysis ... (Done!) \n")
  }

  if(!is.null(control$seed)) set.seed(control$seed)
  fit1 = gapmCDM_fit_rcpp(Y = Ytrain[],A = ppD1$A[],C = ppD1$C[],D = ppD1$D[],mu = ppD1$mu[],R = ppD1$R[], Z = zn[], control = control)
  if(control$return.trace){
    colnames(fit1$cdllk.trace) <- c("fz","fyz")
    Anames <- paste0("A",apply(expand.grid(paste0("j",1:p),paste0("k",1:q)),1,paste,collapse = "."))
    Cnames <- paste0("C",apply(expand.grid(apply(expand.grid(paste0("j",1:p),paste0("r",1:ncol(fit1$C))),1,paste,collapse = "."), paste0("k",1:q)),1,paste,collapse = "."))
    Mnames <- paste0("mu",1:q)
    Rnames <- paste0("R",apply(which(lower.tri(diag(q)) == T,arr.ind = T),1,paste0,collapse = ""))
    colnames(fit1$theta.trace) <- c(Anames,Cnames,Mnames,Rnames)
  }
  if(control$verbose) cat(paste0("\n\n [ Testing data ] \n\n"))

  if(!is.null(control$start.zn.test) && is.matrix(control$start.zn.test)){
    zn.test = control$start.zn.test
  } else if(control$start.zn.test == "random") {
    if(control$verbose) cat(" Generating random starting values for latent variables ...")
    zn.test = mvtnorm::rmvnorm(nrow(Ytest),mean = rep(0,q))
    if(control$verbose) cat("\r Generating random starting values for latent variables ... (Done!) \n")
  } else if(control$start.zn.test == "fa") {
    if(control$verbose) cat(" Generating starting values for latent variables via Factor Analysis ...")
    tmp = suppressMessages(suppressWarnings(psych::fa(r = Ytest, nfactors = q, cor = "tet", fm = "ml", rotate = "oblimin",
                                                      missing = T)))
    zn.test = tmp$scores
    if(control$verbose) cat("\r Generating starting values for latent variables via Factor Analysis ... (Done!) \n")
  }

  testmllk = fy_gapmCDM_IS(Ytest[],A = fit1$A[],C = fit1$C[],mu = fit1$mu[],R = fit1$R[],Z = zn.test[],
                           pM = fit1$posMu[], pR = fit1$posR[], control = control)
  return(list(mllk.test = testmllk, mllk.train = fit1$llk, train.mod = fit1)) #
}

#' Fit an Partial-Mastery Additive Cognitive Diagnosis Model (PM-ACDM).
#'
#' @param data Matrix \code{(n x p)} with binary entries. Missing data should be coded as \code{NA}.
#' @param q Number of latent variables.
#' @param control List of control parameters (see 'Details').
#' @param start.par (optional) For simulation use. List of size 4 with starting model parameters (A, C, D, R).
#' @param ... Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{G}: A matrix \code{(p x (q+1))} of estimated parameters.
#'  \item \code{mu}: An vector \code{(q x 1)} of estimated latent variable means.
#'  \item \code{R}: A matrix \code{(q x q)} of estimated correlations for the latent variables
#'  \item \code{llk}: Marginal log-likelihood (computed via Monte-Carlo integration) evaluated at \code{A}, \code{C}, and \code{R}.
#'  \item \code{AIC}: Akaike information criterion for the estimated model.
#'  \item \code{BIC}: Bayesian (Schwarz) information criterion for the estimated model.
#'  \item \code{cdllk.trace}: (if return.trace = T) A matrix \code{(iter x 2)} with the trace for the observed variables and latent variables log-likelihood.
#'  \item \code{ar.trace}: (if return.trace = T) A vector \code{(iter)} with the trace for the acceptance rate (for MALA and RWMH samplers).
#'  \item \code{theta.trace}: (if return.trace = T) A matrix \code{(iter x dim(theta))} with the trace for the parameter estimates.
#' }
#' @details Test
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
apmCDM <- function(data, q, control = list(), start.par = NULL, ...){

  control = pr_control_aCDM(control,...)
  if(is.null(control$Qmatrix)) control$Qmatrix = matrix(1,p,q)
  if(control$verbose) cat(" Model: Additive PM-CDM \n")
  control$sampler = match.arg(control$sampler, c("ULA","MALA","RWMH"))
  p = ncol(data)
  Apat = as.matrix(expand.grid(lapply(1:q,function(x) c(0,1))))

  if(is.list(start.par)){
    if(length(start.par) != 3) stop("Argument `start.par' needs to be a list with elements `G', `mu', and `R'.")
    pp = start.par
    if(nrow(pp$G) != ncol(data)) stop("Matrix `start.par$G' mismatch rows with p.")
    if(length(pp$mu) != nrow(pp$R)) stop("Lenght of `start.par$mu' and nrows of `start.par$R' differ.")
  }
  if(!is.null(start.par) && (start.par == "random")){
    if(control$verbose) cat(" Generating random starting values for model parameters ...")
    control$iden.R = T
    if(!is.null(control$seed)) set.seed(control$seed)
    pp = pr_param_aCDM(p,q,T,control)
    if(control$verbose) cat("\r Generating random starting values for model parameters ... (Done!) \n")
  }
  if(is.null(start.par)) {
    pp = pr_param_aCDM(p,q,F,control)
  }

  if(!is.null(control$start.zn) && is.matrix(control$start.zn)){
    zn = control$start.zn
  } else if(control$start.zn == "random") {
    if(control$verbose) cat(" Generating random starting values for latent variables ...")
    zn = mvtnorm::rmvnorm(nrow(data),mean = rep(0,q))
    if(control$verbose) cat("\r Generating random starting values for latent variables ... (Done!) \n")
  } else if(control$start.zn == "fa") {
    if(control$verbose) cat(" Generating starting values for latent variables via Factor Analysis ...")
    tmp = suppressWarnings(psych::fa(r = data, nfactors = q, cor = "tet", fm = "ml", rotate = "oblimin",
                                     missing = T))
    zn = tmp$scores
    if(control$cor.R) pp$R = cor(zn) else pp$R = cov(zn)
    if(control$verbose) cat("\r Generating starting values for latent variables via Factor Analysis ... (Done!) \n")
  }

  if(!is.null(control$seed)) set.seed(control$seed)
  out <- apmCDM_fit_rcpp(Y = data[], G = pp$G[], Qmatrix = control$Qmatrix[], Apat = Apat[], mu = pp$mu[], R = pp$R[], Z= zn[], control = control)
  colnames(out$PI) <- rownames(out$G) <- colnames(data)
  colnames(out$G) <- c("(Intercept)",paste0("Z",1:q))
  colnames(out$Z) <- paste0("Z",1:q)
  colnames(out$U) <- paste0("U",1:q)
  if(control$return.trace){
    colnames(out$cdllk.trace) <- c("fz","fyz")
    Gnames <- paste0("G",apply(expand.grid(paste0("j",1:p),paste0("k",0:q)),1,paste,collapse = "."))
    Gnames <- Gnames[which(cbind(1,control$Qmatrix) != 0)]
    Mnames <- paste0("mu",1:q)
    Rnames <- paste0("R",apply(which(lower.tri(diag(q)) == T,arr.ind = T),1,paste0,collapse = ""))
    colnames(out$theta.trace) <- c(Gnames,Mnames,Rnames)
  }
  class(out) = c("apmCDM", "pmCDM")
  return(out)
}


#' Simulate data from an Partial Mastery Additive Cognitive Diagnosis Model (PM-ACDM).
#'
#' @param n Number of simulated entries.
#' @param p Number of observed (binary) variables.
#' @param q Number of latent variables.
#' @param control List of control parameters (see 'Details').
#' @param start.par (optional) List of size 3 with starting model parameters (G, mu, R).
#' @param ... Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{Y}: A matrix \code{(n x p)} of simulated observed (binary) variables.
#'  \item \code{Z}: A matrix \code{(n x q)} of simulated latent variables (on the continuous scale).
#'  \item \code{U}: A matrix \code{(n x q)} of simulated latent variables (on the [0,1] scale).
#'  \item \code{PI}: A matrix \code{(n x p)} of predicted probabilities.
#' }
#' @details Test
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
apmCDM_sim <- function(n, p, q, control = list(),
                        start.par = NULL, ...){
  control = pr_controlsim_aCDM(control,q,...)
  if(is.null(control$Qmatrix)) control$Qmatrix = matrix(1,p,q)
  Apat = as.matrix(expand.grid(lapply(1:q,function(x) c(0,1))))
  if(!is.null(start.par) & !is.list(start.par) & (length(start.par) != 3))
    stop("Argument `start.par' needs to be a list with elements `G', `mu', and `R'.")
  if(!is.null(start.par)){
    pp = start.par
    if(nrow(pp$G) != nrow(control$Qmatrix)) stop("Matrix `start.par$G' mismatch rows with `control$Qmatrix'.")
    if(length(pp$mu) != nrow(pp$R)) stop("Lenght of `start.par$mu' and nrows of `start.par$R' differ.")
  } else {
    pp = pr_param_aCDM(p,q,T,control)
  }
  if(!is.null(control$seed)) set.seed(control$seed)
  out <- apmCDM_sim_rcpp(n,pp$G,control$Qmatrix,Apat,pp$mu,pp$R)
  out$G <- pp$G
  out$mu <- pp$mu
  out$R <- pp$R
  out$posMu <- matrix(colMeans(out$Z),nrow = n, ncol = q, byrow = T)
  out$posR <- array(cov(out$Z),dim = c(q,q,n))
  rownames(out$G) <- colnames(out$Y) <- colnames(out$PI) <- paste0("Y",1:p)
  colnames(out$G) <- c("(Intercept)",paste0("Z",1:q))
  colnames(out$R) <- rownames(out$R) <- colnames(out$posR) <- rownames(out$posR) <- colnames(out$Z) <- names(out$mu) <- names(out$posMu) <-paste0("Z",1:q)
  colnames(out$U) <- paste0("U",1:q)
  class(out) = c("apmCDM", "pmCDM")
  return(out)
}

#' Compare GaPM-CDM vs. PM-aCDM via negative cross-entropy error.
#'
#' @param Ytrain Matrix with observed binary entries to train the model. Missing entries (\code{Ytest}) coded as \code{NA}.
#' @param Ytest Matrix with observed binary entries to test the model. All entries are \code{NA} but the missing in \code{Ytrain}.
#' @param q Number of latent variables.
#' @param controlA List of control parameters for GaPM-CDM (see 'Details').
#' @param controlB List of control parameters for aPM-CDM (see 'Details').
#' @param ... Further arguments to be passed to \code{controlA} and \code{controlB}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{CV.error.A}: Cross-validation error for GaPM-CDM.
#'  \item \code{AUC.A}: Area under the curve error for GaPM-CDM.
#'  \item \code{CV.error.B}: Cross-validation error for aPM-CDM.
#'  \item \code{AUC.B}: Area under the curve error for aPM-CDM.
#' }
#' @details Define CV.error.
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
pmCDM.CV.error <- function(Ytrain, Ytest, q,
                           controlA = list(), controlB = list(), ...){

  controlA = pr_control_gaCDM(controlA,...)
  controlB = pr_control_aCDM(controlB,...)
  controlA$sampler = match.arg(controlA$sampler, c("ULA","MALA","RWMH"))
  controlB$sampler = controlA$sampler
  controlA$basis = match.arg(controlA$basis, c("is","bs","pwl"))
  if(is.null(controlA$degree) && controlA$basis == "pwl") controlA$degree = 1
  if(is.null(controlA$degree) && controlA$basis != "pwl") controlA$degree = 2
  if(is.null(controlA$Qmatrix)) controlA$Qmatrix = matrix(1,p,q)
  if(is.null(controlB$Qmatrix)) controlB$Qmatrix = matrix(1,p,q)

  if(controlA$verbose) cat(paste0("\n Generating random starting values for latent variables q = ", q," ..."))
  zn = mvtnorm::rmvnorm(nrow(Ytrain),mean = rep(0,q))
  if(controlA$verbose) cat(paste0("\r Generating random starting values for latent variables q = ", q," ... (Done!) \n"))

  p = ncol(Ytrain)
  tp = length(controlA$knots) + controlA$degree
  Apat = as.matrix(expand.grid(lapply(1:q,function(x) c(0,1))))
  ppDA = pr_param_gaCDM(p,q,tp,F,controlA)
  ppDB = pr_param_aCDM(p,q,F,controlB)

  if(controlA$verbose) cat(" Fitting model: Generalized Additive PM-CDM \n")
  fitA = gapmCDM_fit_rcpp(Y = Ytrain[],A = ppDA$A[],C = ppDA$C[],D = ppDA$D[],mu = ppDA$mu[], R = ppDA$R[], Z = zn[], control = controlA)
  if(controlA$verbose) cat(" Fitting model: Additive PM-CDM \n")
  fitB = apmCDM_fit_rcpp(Y = Ytrain[],G = ppDB$G[], Qmatrix = controlB$Qmatrix[], Apat = Apat[], mu = ppDB$mu[], R = ppDB$R[], Z = zn[], control = controlB)
  if(controlB$verbose) cat(" CV-Error: Generalized Additive PM-CDM \n")
  outA = gapmCDM_cv_rcpp(Ytrain[], Ytest[],A = fitA$A[],C = fitA$C[],mu = fitA$mu[],R = fitA$R[], Z = zn[], control = controlA)
  if(controlB$verbose) cat(" CV-Error: Additive PM-CDM \n")
  outB = apmCDM_cv_rcpp(Ytrain[], Ytest[],G = fitB$G[],Qmatrix = controlB$Qmatrix[], Apat = Apat[],mu = fitB$mu[],R = fitB$R[], Z = zn[], control = controlB)
  errCVA = outA$CV.error
  errCVB = outB$CV.error
  predA = ROCR::prediction(outA$Yhat,outA$Yobs)
  predB = ROCR::prediction(outB$Yhat,outB$Yobs)
  aucCVA = unlist(methods::slot(ROCR::performance(predA, "auc"), "y.values"))
  aucCVB = unlist(methods::slot(ROCR::performance(predB, "auc"), "y.values"))

  return(list(CV.error.A = errCVA, AUC.A = aucCVA,
              CV.error.B = errCVB, AUC.B = aucCVB)) #
}

#' Find number of latent variables (q) using cross-validation (PM-ACDM)
#'
#' @param Ytrain Matrix with observed binary entries to train the model. Missing entries (\code{Ytest}) coded as \code{NA}.
#' @param Ytest Matrix with observed binary entries to test the model. All entries are \code{NA} but the missing in \code{Ytrain}.
#' @param q Number of latent variables.
#' @param control List of control parameters (see 'Details').
#' @param ... Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{CV.error}: Cross-validation error.
#'  \item \code{AUC}: Area under the curve error.
#'  \item \code{llk.gapmCDM} Marginal log-likelihood for the gapmCDM model
#' }
#' @details Define CV.error.
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
apmCDM_findqCV <- function(Ytrain, Ytest, q, control = list(), start.par = NULL, ...){

  control = pr_control_aCDM(control,...)
  if(control$verbose) cat(" Model: Additive PM-CDM \n")
  control$sampler = match.arg(control$sampler, c("ULA","MALA","RWMH"))
  if(is.null(control$Qmatrix)) control$Qmatrix = matrix(1,p,q)
  p = ncol(data)
  Apat = as.matrix(expand.grid(lapply(1:q,function(x) c(0,1))))

  if(is.list(start.par)){
    if(length(start.par) != 3) stop("Argument `start.par' needs to be a list with elements `G', `mu', and `R'.")
    ppD1 = start.par
    if(nrow(ppD1$G) != ncol(data)) stop("Matrix `start.par$G' mismatch rows with p.")
    if(length(ppD1$mu) != nrow(ppD1$R)) stop("Lenght of `start.par$mu' and nrows of `start.par$R' differ.")
  }
  if(!is.null(start.par) && (start.par == "random")){
    if(control$verbose) cat(" Generating random starting values for model parameters ...")
    control$iden.R = T
    if(!is.null(control$seed)) set.seed(control$seed)
    ppD1 = pr_param_aCDM(p,q,T,control)
    if(control$verbose) cat("\r Generating random starting values for model parameters ... (Done!) \n")
  }
  if(is.null(start.par)) {
    ppD1 = pr_param_aCDM(p,q,F,control)
  }

  if(!is.null(control$start.zn) && is.matrix(control$start.zn)){
    zn = control$start.zn
  } else if(control$start.zn == "random") {
    if(control$verbose) cat(" Generating random starting values for latent variables ...")
    zn = mvtnorm::rmvnorm(nrow(data),mean = rep(0,q))
    if(control$verbose) cat("\r Generating random starting values for latent variables ... (Done!) \n")
  } else if(control$start.zn == "fa") {
    if(control$verbose) cat(" Generating starting values for latent variables via Factor Analysis ...")
    tmp = suppressWarnings(psych::fa(r = data, nfactors = q, cor = "tet", fm = "ml", rotate = "oblimin",
                                     missing = T))
    zn = tmp$scores
    if(control$cor.R) ppD1$R = cor(zn) else ppD1$R = cov(zn)
    if(control$verbose) cat("\r Generating starting values for latent variables via Factor Analysis ... (Done!) \n")
  }

  if(!is.null(control$seed)) set.seed(control$seed)
  if(control$verbose) cat(paste0("\n [Training data] \n"))
  fit1 = apmCDM_fit_rcpp(Y = Ytrain[], G = ppD1$G[], Qmatrix = control$Qmatrix[], Apat = Apat[], mu = ppD1$mu[], R = ppD1$R[], Z= zn[], control = control)
  out1 = apmCDM_cv_rcpp(Ytrain[], Ytest[], G = fit1$G[], Qmatrix = control$Qmatrix, Apat = Apat[], mu = fit1$mu[],R = fit1$R[], Z = fit1$Z[], control = control)
  pred = ROCR::prediction(out1$Yhat,out1$Yobs)
  aucCV = unlist(methods::slot(ROCR::performance(pred, "auc"), "y.values"))

  return(list(CV.error = out1$CV.error, AUC = aucCV, llk.train = fit1$llk,
              mod.train = fit1))
}


#' Find number of latent variables (q) using cross-validated MLLK (PM-ACDM)
#'
#' @param Ytrain Matrix with observed binary entries to train the model. Missing entries (\code{Ytest}) coded as \code{NA}.
#' @param Ytest Matrix with observed binary entries to test the model. All entries are \code{NA} but the missing in \code{Ytrain}.
#' @param q Number of latent variables.
#' @param control List of control parameters (see 'Details').
#' @param ... Further arguments to be passed to \code{control}.
#'
#' @return A list with components:
#' \itemize{
#'  \item \code{mllk.test}: Test data marginal log-likelihood.
#'  \item \code{mllk.train}: Train data marginal log-likelihood.
#'  \item \code{mllk.BIC} Train data BIC.
#' }
#' @details Define CV.error.
#' @author Camilo Cárdenas-Hurtado (\email{c.a.cardenas-hurtado@@lse.ac.uk}).
#' @export
apmCDM_mllkCV <- function(Ytrain, Ytest, q, control = list(), ...){
  control = pr_control_aCDM(control,...)
  if(control$verbose) cat(" Model: Additive PM-CDM \n")
  if(control$verbose) cat(paste0("\n [ Training data ] \n\n"))
  control$sampler = match.arg(control$sampler, c("ULA","MALA","RWMH"))
  if(is.null(control$Qmatrix)) control$Qmatrix = matrix(1,p,q)

  p = ncol(Ytrain)
  ppD1 = pr_param_aCDM(p,q,F,control)

  if(!is.null(control$start.zn) && is.matrix(control$start.zn)){
    zn = control$start.zn
  } else if(control$start.zn == "random") {
    if(control$verbose) cat(" Generating random starting values for latent variables ...")
    zn = mvtnorm::rmvnorm(nrow(Ytrain),mean = rep(0,q))
    if(control$verbose) cat("\r Generating random starting values for latent variables ... (Done!) \n")
  } else if(control$start.zn == "fa") {
    if(control$verbose) cat(" Generating starting values for latent variables via Factor Analysis ...")
    tmp = suppressMessages(suppressWarnings(psych::fa(r = Ytrain, nfactors = q, cor = "tet", fm = "ml", rotate = "oblimin",
                                                      missing = T)))
    zn = tmp$scores
    if(control$cor.R) ppD1$R = cor(zn) else ppD1$R = cov(zn)
    if(control$verbose) cat("\r Generating starting values for latent variables via Factor Analysis ... (Done!) \n")
  }

  Apat = as.matrix(expand.grid(lapply(1:q,function(x) c(0,1))))
  if(!is.null(control$seed)) set.seed(control$seed)
  fit1 = apmCDM_fit_rcpp(Y = Ytrain[], G = ppD1$G[], Qmatrix = control$Qmatrix[], Apat = Apat[], mu = ppD1$mu[], R = ppD1$R[], Z= zn[], control = control)
  if(control$return.trace){
    colnames(fit1$cdllk.trace) <- c("fz","fyz")
    Gnames <- paste0("G",apply(expand.grid(paste0("j",1:p),paste0("k",0:q)),1,paste,collapse = "."))
    Gnames <- Gnames[which(cbind(1,control$Qmatrix) != 0)]
    Mnames <- paste0("mu",1:q)
    Rnames <- paste0("R",apply(which(lower.tri(diag(q)) == T,arr.ind = T),1,paste0,collapse = ""))
    colnames(fit1$theta.trace) <- c(Gnames,Mnames,Rnames)
  }

  if(control$verbose) cat(paste0("\n\n [ Testing data ] \n\n"))

  if(!is.null(control$start.zn.test) && is.matrix(control$start.zn.test)){
    zn.test = control$start.zn.test
  } else if(control$start.zn.test == "random") {
    if(control$verbose) cat(" Generating random starting values for latent variables ...")
    zn.test = mvtnorm::rmvnorm(nrow(Ytest),mean = rep(0,q))
    if(control$verbose) cat("\r Generating random starting values for latent variables ... (Done!) \n")
  } else if(control$start.zn.test == "fa") {
    if(control$verbose) cat(" Generating starting values for latent variables via Factor Analysis ...")
    tmp = suppressMessages(suppressWarnings(psych::fa(r = Ytest, nfactors = q, cor = "tet", fm = "ml", rotate = "oblimin",
                                                      missing = T)))
    zn.test = tmp$scores
    if(control$verbose) cat("\r Generating starting values for latent variables via Factor Analysis ... (Done!) \n")
  }

  testmllk = fy_aCDM_IS(Ytest[],G = fit1$G[],Qmatrix = control$Qmatrix[], Apat = Apat[],
                        mu = fit1$mu[],R = fit1$R[], Z = zn.test[],
                        pM = fit1$posMu[], pR = fit1$posR[], control = control)
  return(list(mllk.test = testmllk, mllk.train = fit1$llk, train.mod = fit1)) #
}
