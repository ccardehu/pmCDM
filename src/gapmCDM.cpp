#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

#include <RcppArmadillo.h>
#include "utils.h"
#include "estim.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List gapmCDM_sim_rcpp(const int n, const int q, const int p,
                            arma::mat& A, arma::cube& C,
                            arma::vec& mu, arma::mat& R,
                            Rcpp::List& control){
  arma::vec knots = control["knots"];
  const unsigned int degree = control["degree"];
  const std::string basis = Rcpp::as<std::string>(control["basis"]);
  arma::mat Y(n,p);
  arma::mat Z = rmvNorm(n,mu,R);
  arma::mat U = Z2U(Z);
  arma::mat spM(n,q*C.n_cols);
  arma::cube spObj(n,q*C.n_cols,2);
  if(basis == "is"){
    spObj = SpU_isp(U,knots,degree);
    spM = spObj.slice(0);
  } else {
    spObj = SpU_bsp(U,knots,degree);
    spM = spObj.slice(0);
  }
  arma::mat PI = prob(A,C,spM);
  for(int j = 0; j < p; j++){
    for(int i = 0; i < n; i++){
      Y(i,j) = R::rbinom(1,PI(i,j));
    }
  }
  return Rcpp::List::create(Rcpp::Named("Y") = Y,
                            Rcpp::Named("Z") = Z,
                            Rcpp::Named("U") = U,
                            Rcpp::Named("spM") = spM,
                            Rcpp::Named("PI") = PI) ;
}

// [[Rcpp::export]]
Rcpp::List gapmCDM_fit_rcpp(arma::mat& Y, arma::mat& A, arma::cube& C, arma::cube& D,
                            arma::vec& mu, arma::mat& R, arma::mat& Z, Rcpp::List& control){

  const int burnin = control["burn.in"];
  const int iterlim = control["iter.lim"];
  const int tunelim = control["tune.lim"];
  const int window = control["window"];
  const unsigned int degree = control["degree"];
  const double h = control["h"];
  const double epsStop = control["stop.eps"];
  const double epsTune = control["tune.eps"];
  const bool verbose = control["verbose"];
  const bool stopFLAG = control["stop.atconv"];
  const bool traceFLAG = control["return.trace"];
  const std::string sampler = Rcpp::as<std::string>(control["sampler"]);
  const std::string basis = Rcpp::as<std::string>(control["basis"]);
  arma::vec knots = control["knots"];

  const int n = Y.n_rows;
  const int q = R.n_cols;
  const int np = knots.size() + degree ;

  arma::mat L = arma::chol(R,"lower");
  arma::mat PI(arma::size(Y));
  arma::mat Zout(arma::size(Z));
  arma::mat Aout(arma::size(A)), An(arma::size(A));
  arma::cube Cout(arma::size(C)), Cn(arma::size(C));
  arma::cube Dout(arma::size(D)), Dn(arma::size(D));
  arma::vec Mout(arma::size(mu)), Mn(arma::size(mu));
  arma::mat Rout(arma::size(R));
  arma::mat spM(n,q*np), spD(n,q*np);
  arma::cube spObj(n,q*np,2);

  arma::uvec Rld = arma::trimatl_ind(arma::size(R),-1);

  const int tp = A.size() + C.size() + mu.size() + Rld.size();
  arma::mat theta(window,tp);
  arma::mat patrace(iterlim/10,tp);
  arma::mat lltrace(iterlim/10,2);
  arma::vec artrace(iterlim/10);
  int iter = 1;
  double ar = 0;

  for(int ii = 1; ii <= iterlim + tunelim; ii++){
    if (ii % 2 == 0) Rcpp::checkUserInterrupt();
    arma::mat U = Z2U(Z);
    double ssAC;
    double ssZ;
    if(ii <= tunelim){
      ssAC = 0.0 + epsTune ;
      if(verbose & (ii % 10 == 0)) Rcpp::Rcout << "\r Iteration: " << std::setw(5) << ii << " (tuning phase)";
      ar = 0 ;
      ssZ = 0.0 + 1.0*h ;
    } else {
      ssAC = std::pow(iter,-.833) ;
      if(sampler == "ULA"){
        ssZ = 0.0 + 1.0 * h * std::pow(iter,-.333) ; // * std::pow(q,-.333)
      } else if(sampler == "MALA"){
        ssZ = 0.0 + 1.0 * h * std::pow(iter,-.333) ; // * std::pow(q,-.333)
      } else if(sampler == "RWMH"){
        ssZ = 1.0 * h * std::pow(q,-1) ;
      }
    }
    if(basis == "is"){
      spObj = SpU_isp(U,knots,degree);
      spM = spObj.slice(0);
      spD = spObj.slice(1);
      PI = prob(A,C,spM);
      Rcpp::List gac = d1AC(Y,PI,spM,A,C);
      Rcpp::List ACn = newAC_MD(gac,A,C,ssAC/n);
      An = Rcpp::as<arma::mat>(ACn["A"]);
      Cn = Rcpp::as<arma::cube>(ACn["C"]);
    } else {
      spObj = SpU_bsp(U,knots,degree);
      spM = spObj.slice(0);
      spD = spObj.slice(1);
      PI = prob(A,C,spM);
      Rcpp::List gad = d1AD(Y,PI,spM,A,D);
      Rcpp::List ADn = newAD_MD(gad,A,D,ssAC/n);
      An = Rcpp::as<arma::mat>(ADn["A"]);
      Dn = Rcpp::as<arma::cube>(ADn["D"]);
      Cn = D2C(Dn);
    }
    // arma::vec Mn = newM(mu,R,Z,ssAC/n);
    arma::mat Ln = newL(mu,L,Z,ssAC/n);
    arma::mat Zn(arma::size(Z));
    if(sampler == "ULA"){
      Zn = newZ_ULA(Y,PI,Z,A,C,mu,R,spD,ssZ,knots,degree);
    } else if(sampler == "MALA"){
      Zn = newZ_MALA(Y,PI,Z,A,C,mu,R,ar,spD,ssZ,knots,degree,basis);
    } else if(sampler == "RWMH"){
      Zn = newZ_RWMH(Y,PI,Z,A,C,mu,R,ar,h,knots,degree,basis);
    }
    A = An;
    C = Cn;
    D = Dn;
    // mu = Mn;
    L = Ln;
    R = Ln*Ln.t();
    Z = Zn;
    if(iter >= burnin){
      Aout += A;
      Cout += C;
      Dout += D;
      Mout += mu;
      Rout += R;
      Zout += Z;
      for(int t1 = window - 1; t1 > 0; t1--){
        theta.row(t1) = theta.row(t1 - 1);
      }
      arma::rowvec ai = arma::vectorise(A).t();
      arma::rowvec ci = arma::vectorise(C).t();
      arma::rowvec mi = mu.t();
      arma::rowvec ri = R(Rld).t();
      arma::rowvec input = arma::join_rows(ai,ci,mi,ri);
      theta.row(0) = input;
      arma::mat dtheta = arma::diff(theta,1,0);
      double maxvalue = arma::max(arma::abs(arma::vectorise(dtheta)));
      if(iter % 10 == 0){
        patrace.row(iter/10-1) = input;
        double fzll = arma::accu(fz(Z,R));
        lltrace(iter/10-1,0) = fzll;
        double fyzll = arma::accu(fyz(Y,PI));
        lltrace(iter/10-1,1) = fyzll;
        double cdll = fyzll + fzll;
        artrace(iter/10-1) = ar/(n*iter);
        if(verbose){
          if(sampler == "ULA"){
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (estimation phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ")";
          } else {
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (estimation phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ", AR (" << sampler << "): " << std::to_string(ar/(n*iter)) << ")";
          }
        }
        if(stopFLAG & (maxvalue < epsStop)) {
          iter++;
          break;
        }
      }
    } else {
      if(verbose & (iter % 10 == 0)) Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (burn-in phase)";
    }
    if(ii > tunelim) iter++;
  }
  Aout /= (iter - burnin);
  Cout /= (iter - burnin);
  Dout /= (iter - burnin);
  Mout /= (iter - burnin);
  Rout /= (iter - burnin);
  Zout /= (iter - burnin);

  const int nsim = control["nsim"];
  double llk(0), BIC(0), AIC(0);
  if(nsim != 0){
    llk = fy(Y,Aout,Cout,Mout,Rout,control);
    BIC = -2*llk + std::log(n)*Aout.n_elem;
    AIC = -2*llk + 2*Aout.n_elem;
  }

  if(traceFLAG){
    return Rcpp::List::create(Rcpp::Named("A") = Aout,
                              Rcpp::Named("C") = Cout,
                              Rcpp::Named("mu") = Mout,
                              Rcpp::Named("R") = Rout,
                              Rcpp::Named("Z") = Zout,
                              Rcpp::Named("llk") = llk,
                              Rcpp::Named("BIC") = BIC,
                              Rcpp::Named("AIC") = AIC,
                              Rcpp::Named("cdllk.trace") = lltrace,
                              Rcpp::Named("ar.trace") = artrace,
                              Rcpp::Named("theta.trace") = patrace);
  } else {
    return Rcpp::List::create(Rcpp::Named("A") = Aout,
                              Rcpp::Named("C") = Cout,
                              Rcpp::Named("mu") = Mout,
                              Rcpp::Named("R") = Rout,
                              Rcpp::Named("Z") = Zout,
                              Rcpp::Named("llk") = llk,
                              Rcpp::Named("BIC") = BIC,
                              Rcpp::Named("AIC") = AIC);
  }
}


// [[Rcpp::export]]
Rcpp::List gapmCDM_cv_rcpp(arma::mat& Ytrain, arma::mat& Ytest, arma::mat& A, arma::cube& C,
                           arma::vec& mu, arma::mat& R, arma::mat& Z, Rcpp::List& control){

  const int nsim = control["nsim"];
  const unsigned int degree = control["degree"];
  const double h = control["h"];
  arma::vec knots = control["knots"];
  const bool verbose = control["verbose"];
  const std::string sampler = Rcpp::as<std::string>(control["sampler"]);
  const std::string basis = Rcpp::as<std::string>(control["basis"]);

  const int n = Ytrain.n_rows;
  const int p = Ytrain.n_cols;
  const int q = R.n_cols;
  const int np = knots.size() + degree ;

  arma::mat piH(n,p);
  arma::uvec isNA = arma::find_nan(Ytrain);
  const int nmis = isNA.n_elem;

  arma::mat spM(n,q*np), spD(n,q*np);
  arma::cube spObj(n,q*np,2);

  for(int ii = 1; ii <= nsim; ii++){
    if (ii % 2 == 0) Rcpp::checkUserInterrupt();
    arma::mat U = Z2U(Z);
    if(basis == "is"){
      spObj = SpU_isp(U,knots,degree);
      spM = spObj.slice(0);
      spD = spObj.slice(1);
    } else {
      spObj = SpU_bsp(U,knots,degree);
      spM = spObj.slice(0);
      spD = spObj.slice(1);
    }
    arma::mat PI = prob(A,C,spM);
    arma::mat Zn(arma::size(Z));
    double ar = 0;
    double ssZ = h * std::pow(ii,-.33) ;
    if(sampler == "ULA"){
      Zn = newZ_ULA(Ytrain,PI,Z,A,C,mu,R,spD,ssZ,knots,degree);
    } else if(sampler == "MALA"){
      Zn = newZ_MALA(Ytrain,PI,Z,A,C,mu,R,ar,spD,ssZ,knots,degree,basis);
    } else if(sampler == "RWMH"){
      Zn = newZ_RWMH(Ytrain,PI,Z,A,C,mu,R,ar,h,knots,degree,basis);
    }
    piH += PI;
    Z = Zn;
    if(verbose & (ii % 10 == 0)) Rcpp::Rcout << "\r CV iteration: " << ii ;
  }
  piH /= nsim;
  double out = 0;
  for(int ii = 0; ii < nmis; ii++){
    arma::uword id = isNA(ii);
    out += Ytest(id)*std::max(arma::datum::log_min, std::log(piH(id))) + (1-Ytest(id))*std::max(arma::datum::log_min, std::log(1-piH(id)));
  }
  out /= -nmis;
  if(verbose) Rcpp::Rcout << "\r CV iteration: " << nsim << " ... [CV-error: " << std::to_string(out) << "]\n" ;
  arma::vec outYhat = piH(isNA);
  arma::vec outYtst = Ytest(isNA);
  return Rcpp::List::create(Rcpp::Named("Yhat") = Rcpp::wrap(outYhat),
                            Rcpp::Named("Yobs") = Rcpp::wrap(outYtst),
                            Rcpp::Named("CV.error") = out);
}

// [[Rcpp::export]]
Rcpp::List apmCDM_sim_rcpp(const int n,
                         arma::mat& G, arma::mat& Qmatrix, arma::mat& Apat,
                         arma::vec& mu, arma::mat& R){
  int p = G.n_rows;
  arma::mat Y(n,p);
  arma::mat Z = rmvNorm(n,mu,R);
  arma::mat U = Z2U(Z);
  Rcpp::List aCDMlist = aCDM(G,Qmatrix,Z,Apat);
  arma::mat PI = aCDMlist["PI"];
  for(int j = 0; j < p; j++){
    for(int i = 0; i < n; i++){
      Y(i,j) = R::rbinom(1,PI(i,j));
    }
  }
  return Rcpp::List::create(Rcpp::Named("Y") = Y,
                            Rcpp::Named("Z") = Z,
                            Rcpp::Named("U") = U,
                            Rcpp::Named("PI") = PI) ;
}

// [[Rcpp::export]]
Rcpp::List apmCDM_fit_rcpp(arma::mat& Y, arma::mat& G, arma::mat& Qmatrix, arma::mat& Apat,
                           arma::vec& mu, arma::mat& R, arma::mat& Z, Rcpp::List& control){

  const int burnin = control["burn.in"];
  const int iterlim = control["iter.lim"];
  const int tunelim = control["tune.lim"];
  const int window = control["window"];
  const double h = control["h"];
  const double epsStop = control["stop.eps"];
  const double epsTune = control["tune.eps"];
  const bool verbose = control["verbose"];
  const bool stopFLAG = control["stop.atconv"];
  const bool traceFLAG = control["return.trace"];
  const std::string sampler = Rcpp::as<std::string>(control["sampler"]);

  const int n = Y.n_rows;
  const int p = Y.n_cols;
  const int q = R.n_cols;

  arma::mat L = arma::chol(R,"lower");
  arma::mat PI(arma::size(Y));
  arma::mat Zout(arma::size(Z));
  arma::mat Gout(arma::size(G)), Gn(arma::size(G));
  arma::vec Mout(arma::size(mu)), Mn(arma::size(mu));
  arma::mat Rout(arma::size(R));

  arma::uvec Rld = arma::trimatl_ind(arma::size(R),-1);

  arma::vec c1(p, arma::fill::value(1.0));
  arma::mat Qmatrix1 = arma::join_rows(c1,Qmatrix);
  arma::uvec iG = arma::find(Qmatrix1 != 0);

  const int tp = iG.size() + mu.size() + Rld.size();
  arma::mat theta(window,tp);
  arma::mat patrace(iterlim/10,tp);
  arma::mat lltrace(iterlim/10,2);
  arma::vec artrace(iterlim/10);
  int iter = 1;
  double ar = 0;

  for(int ii = 1; ii <= iterlim + tunelim; ii++){
    if (ii % 2 == 0) Rcpp::checkUserInterrupt();
    arma::mat U = Z2U(Z);
    double ssG;
    double ssZ;
    if(ii <= tunelim){
      ssG = 0.0 + epsTune ;
      if(verbose & (ii % 10 == 0)) Rcpp::Rcout << "\r Iteration: " << std::setw(5) << ii << " (tuning phase)";
      ar = 0 ;
      ssZ = 0.0 + 1.0*h ;
    } else {
      ssG = std::pow(iter,-.833) ;
      if(sampler == "ULA"){
        ssZ = 0.0 + 1.0 * h * std::pow(iter,-.333) ; // * std::pow(q,-.333)
      } else if(sampler == "MALA"){
        ssZ = 0.0 + 1.0 * h * std::pow(iter,-.333) ; // * std::pow(q,-.333)
      } else if(sampler == "RWMH"){
        ssZ = 1.0 * h * std::pow(q,-1) ;
      }
    }
    Rcpp::List aCDMlist = aCDM(G, Qmatrix, Z, Apat);
    PI = Rcpp::as<arma::mat>(aCDMlist["PI"]);
    Rcpp::List dG = d1G(Y, aCDMlist);
    arma::mat Gn = newG_MD(dG,G,ssG/n);
    arma::vec Mn = newM(mu,R,Z,ssG/n);
    arma::mat Ln = newL(mu,L,Z,ssG/n);
    arma::mat Zn(arma::size(Z));
    if(sampler == "ULA"){
      Zn = newZ_ULA_aCDM(Y,Z,aCDMlist,mu,R,ssZ);
    } else if(sampler == "MALA"){
      Zn = newZ_MALA_aCDM(Y,Z,aCDMlist,G,Qmatrix,Apat,mu,R,ssZ,ar);
    } else if(sampler == "RWMH"){
      Zn = newZ_RWMH_aCDM(Y,Z,aCDMlist,G,Qmatrix,Apat,mu,R,h,ar);
    }
    G = Gn;
    mu = Mn;
    L = Ln;
    R = Ln*Ln.t();
    Z = Zn;
    if(iter >= burnin){
      Gout += G;
      Mout += mu;
      Rout += R;
      Zout += Z;
      for(int t1 = window - 1; t1 > 0; t1--){
        theta.row(t1) = theta.row(t1 - 1);
      }
      arma::vec Gpar = G(iG);
      arma::rowvec gi = Gpar.t();
      arma::rowvec mi = mu.t();
      arma::rowvec ri = R(Rld).t();
      arma::rowvec input = arma::join_rows(gi,mi,ri);
      theta.row(0) = input;
      arma::mat dtheta = arma::diff(theta,1,0);
      double maxvalue = arma::max(arma::abs(arma::vectorise(dtheta)));
      if(iter % 10 == 0){
        patrace.row(iter/10-1) = input;
        double fzll = arma::accu(fz(Z,R));
        lltrace(iter/10-1,0) = fzll;
        double fyzll = arma::accu(fyz(Y,PI));
        lltrace(iter/10-1,1) = fyzll;
        double cdll = fyzll + fzll;
        artrace(iter/10-1) = ar/(n*iter);
        if(verbose){
          if(sampler == "ULA"){
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (estimation phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ")";
          } else {
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (estimation phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ", AR: " << std::to_string(ar/(n*iter)) << ")";
          }
        }
        if(stopFLAG & (maxvalue < epsStop)) {
          iter++;
          break;
        }
      }
    } else {
      if(verbose & (iter % 10 == 0)) Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (burn-in phase)";
    }
    if(ii > tunelim) iter++;
  }
  Gout /= (iter - burnin);
  Mout /= (iter - burnin);
  Rout /= (iter - burnin);
  Zout /= (iter - burnin);

  const int nsim = control["nsim"];
  double llk(0), BIC(0), AIC(0);
  if(nsim != 0){
    llk = fy_aCDM(Y,Gout,Qmatrix,Apat,Mout,Rout,control);
    BIC = -2*llk + std::log(n)*tp;
    AIC = -2*llk + 2*tp;
  }

  if(traceFLAG){
    return Rcpp::List::create(Rcpp::Named("G") = Gout,
                              Rcpp::Named("mu") = Mout,
                              Rcpp::Named("R") = Rout,
                              Rcpp::Named("Z") = Zout,
                              Rcpp::Named("llk") = llk,
                              Rcpp::Named("BIC") = BIC,
                              Rcpp::Named("AIC") = AIC,
                              Rcpp::Named("cdllk.trace") = lltrace,
                              Rcpp::Named("ar.trace") = artrace,
                              Rcpp::Named("theta.trace") = patrace);
  } else {
    return Rcpp::List::create(Rcpp::Named("G") = Gout,
                              Rcpp::Named("mu") = Mout,
                              Rcpp::Named("R") = Rout,
                              Rcpp::Named("Z") = Zout,
                              Rcpp::Named("llk") = llk,
                              Rcpp::Named("BIC") = BIC,
                              Rcpp::Named("AIC") = AIC);
  }
}
