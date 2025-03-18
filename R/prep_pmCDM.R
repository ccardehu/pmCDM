pr_control_gaCDM <- function(control, ...){

  con <- list("burn.in" = 3e3, "iter.lim" = 1e4, "tune.lim" = 2e3,
              "stop.eps" = 1e-5, "gamma" = 1,
              "h" = 1e-2, "tune.eps" = 1, "return.trace" = F,
              "degree" = NULL, "knots" = seq(.1,.9,by = 0.1),"cor.R" = T,
              "nsim" = 1000, "verbose" = T, "seed" = NULL, "mu" = NULL, "R" = NULL,
              "start.zn" = "random", "sampler" = "ULA", "basis" = "pwl",
              "window" = 10, "stop.atconv" = T)
  control <- c(control, list(...))
  namC <- names(con)
  con[(namc <- names(control))] <- control
  if (length(namc[!namc %in% namC]) > 0)
    warning("Unknown names in control: ", paste(namc[!namc %in% namC], collapse = ", "))
  return(con)
}

pr_control_aCDM <- function(control, ...){

  con <- list("burn.in" = 3e3, "iter.lim" = 1e4, "tune.lim" = 2e3,
              "stop.eps" = 1e-5, "gamma" = 1,
              "h" = 1e-2, "tune.eps" = 1, "return.trace" = F,
              "max.G0" = 0.10, "cor.R" = T,
              "nsim" = 1000, "verbose" = T, "seed" = NULL, "mu" = NULL, "R" = NULL,
              "start.zn" = "random", "sampler" = "ULA",
              "window" = 10, "stop.atconv" = T)
  control <- c(control, list(...))
  namC <- names(con)
  con[(namc <- names(control))] <- control
  if (length(namc[!namc %in% namC]) > 0)
    warning("Unknown names in control: ", paste(namc[!namc %in% namC], collapse = ", "))
  return(con)
}

pr_controlsim_gaCDM <- function(control,q,...){

  con <- list("degree" = NULL, "knots" = seq(.1,.9,by=0.2), "prob.sparse" = 0.75,
              "iden.R" = F, "seed" = NULL, "mu" = NULL, "R" = NULL, "basis" = "is")
  control <- c(control, list(...))
  namC <- names(con)
  con[(namc <- names(control))] <- control
  if (length(namc[!namc %in% namC]) > 0)
    warning("Unknown names in control: ", paste(namc[!namc %in% namC], collapse = ", "))
  return(con)
}

pr_controlsim_aCDM <- function(control,q,...){

  con <- list("iden.R" = F, "max.G0" = 0.10, "seed" = NULL, "mu" = NULL, "R" = NULL)
  control <- c(control, list(...))
  namC <- names(con)
  con[(namc <- names(control))] <- control
  if (length(namc[!namc %in% namC]) > 0)
    warning("Unknown names in control: ", paste(namc[!namc %in% namC], collapse = ", "))
  return(con)
}

#' @export
pr_param_gaCDM <- function(p,q,tp,sim = F,control){
  if(!sim){
      As <- matrix(1/q,p,q)
      Ds <- array(stats::rnorm(p*tp*q), dim = c(p,tp,q))
      if(control$basis == "is"){
        Cs <- array(1/tp,dim = c(p,tp,q))
      } else {
        Cs <- D2C(Ds)
      }
      Rs <- diag(q)
      mu <- rep(0,q)
      return(list("A" = As, "C" = Cs, "D" = Ds, "mu" = mu, "R" = Rs))
  } else {
    simp <- genpar(p,q,control$prob.sparse,control$knots,control$degree,control$basis)
    As <- simp$A
    Cs <- simp$C
    Ds <- simp$D
    if(is.null(control$R)){
      if(control$iden.R){
        control$R = diag(q)
      } else {
        control$R = simstudy::genCorMat(q, cors = seq(-0.5,0.5,length.out = q*(q-1)/2))
      }
    }
    if(is.null(control$mu)){
      control$mu = rep(0,q)
    }
    return(list("A" = As, "C" = Cs, "D" = Ds, "mu" = control$mu, "R" = control$R))
  }
}

#' @export
pr_param_aCDM <- function(p,q,Qmatrix,sim = F,control){
  if(!sim){
    Gs <- genpar_aCDM(Qmatrix, control$max.G0)
    Rs <- diag(q)
    mu <- rep(0,q)
    return(list("G" = Gs, "mu" = mu, "R" = Rs))
  } else {
    Gs <- genpar_aCDM(Qmatrix, control$max.G0)
    if(is.null(control$R)){
      if(control$iden.R){
        control$R = diag(q)
      } else {
        control$R = simstudy::genCorMat(q, cors = seq(-0.5,0.5,length.out = q*(q-1)/2))
      }
    }
    if(is.null(control$mu)){
      control$mu = rep(0,q)
    }
    return(list("G" = Gs, "mu" = control$mu, "R" = control$R))
  }
}
