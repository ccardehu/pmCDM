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
  const double epsStop = control["stop.eps"];
  const double epsTune = control["tune.eps"];
  const bool verbose = control["verbose"];
  const bool stopFLAG = control["stop.atconv"];
  const bool traceFLAG = control["return.trace"];
  const bool corFLAG = control["cor.R"];
  const std::string sampler = Rcpp::as<std::string>(control["sampler"]);
  const std::string basis = Rcpp::as<std::string>(control["basis"]);
  const std::string algo = Rcpp::as<std::string>(control["algorithm"]);
  double h = control["h"];
  double g = control["gamma"];
  arma::vec knots = control["knots"];

  const int n = Y.n_rows;
  const int q = R.n_cols;
  const int np = knots.size() + degree ;

  arma::mat L = arma::chol(R,"lower");
  arma::mat PI(arma::size(Y));
  arma::mat Zout(arma::size(Z));
  arma::mat Aout(arma::size(A)), An(A);
  arma::cube Cout(arma::size(C)), Cn(arma::size(C));
  arma::cube Dout(arma::size(D)), Dn(arma::size(D));
  arma::vec Mout(arma::size(mu)), Mn(arma::size(mu));
  arma::mat Rout(arma::size(R));
  arma::mat spM(n,q*np), spD(n,q*np);
  arma::cube spObj(n,q*np,2);
  // arma::mat posR(q,q);
  arma::cube pR(q,q,n), pRn(q,q,n), pRout(q,q,n);
  cube2eye(pR);
  // arma::rowvec posM(q);
  arma::mat pM(arma::size(Z)), pMn(arma::size(Z)), pMout(arma::size(Z));

  arma::uvec Rld = arma::trimatl_ind(arma::size(R),-1);

  const int tp = A.size() + C.size() + mu.size() + Rld.size();
  arma::mat theta(window,tp);
  arma::mat patrace(iterlim/10,tp);
  arma::mat lltrace(iterlim/10,2);
  arma::vec artrace(iterlim/10);
  const int etp = A.size() + C.size();
  arma::vec mt(etp,fill::zeros), vt(etp,fill::zeros);
  // gs(etp), vs(etp,fill::zeros),
  // arma::vec vS(etp,fill::zeros); //
  // arma::mat hS(etp,etp);
  int iter = 1;
  double ar = 0;

  for(int ii = 1; ii <= iterlim + tunelim; ii++){
    if (ii % 2 == 0) Rcpp::checkUserInterrupt();
    arma::mat U = Z2U(Z);
    double ssAC;
    double ssZ;
    if(ii <= tunelim){
      ssAC = epsTune/n ;
      if(verbose & (ii % 10 == 0)) Rcpp::Rcout << "\r Iteration: " << std::setw(5) << ii << " (tuning phase)";
      ar = 0 ;
      ssZ = h ;
    } else {
      // ssAC = g/n * std::pow(iter,-.833) ;
      ssAC = g/n * std::pow(iter,-.51) ;
      if(sampler == "ULA"){
        ssZ = h * std::pow(iter,-.333) ;
      } else if(sampler == "MALA"){
        ssZ = h * std::pow(q,-.333) ;
      } else if(sampler == "RWMH"){
        ssZ = h * std::pow(q,-1) ;
      }
    }
    if(basis == "is"){
      spObj = SpU_isp(U,knots,degree);
      spM = spObj.slice(0);
      spD = spObj.slice(1);
      PI = prob(A,C,spM);
      Rcpp::List gac = d1AC(Y,PI,spM,A,C);
      Rcpp::List ACn = newAC_MD(gac,A,C,ssAC);
      An = Rcpp::as<arma::mat>(ACn["A"]);
      Cn = Rcpp::as<arma::cube>(ACn["C"]);
    } else {
      spObj = SpU_bsp(U,knots,degree);
      spM = spObj.slice(0);
      spD = spObj.slice(1);
      PI = prob(A,C,spM);
      Rcpp::List gad = d1AD(Y,PI,spM,A,D);
      // arma::vec gs = Rcpp::as<arma::vec>(gad["gs"]);
      // arma::mat hS = Rcpp::as<arma::mat>(gad["hs"]);
      // arma::vec vs = arma::diagvec(hS);
      // vs1 = (1-ssAC)*vs1 + ssAC*(vs - arma::pow(gs,2));
      // vs2 = (1-ssAC)*vs2 + ssAC*(gs);
      // arma::vec vSt = vs1 + arma::pow(vs2,2);
      // vS = (ii-1)/ii*vS + 1/ii*arma::clamp(vSt,1e-3,1e3);
      Rcpp::List ADn ;
      if(algo == "GD"){
        ADn = newAD_MD(gad,A,D,ssAC);
      } else if(algo == "ADAM"){
        ADn = newAD_MD_adam(gad,A,D,ssAC,iter,mt,vt,control);
      } else if(algo == "mixed"){
        if(ii <= tunelim + 0.75*iterlim){
          ADn = newAD_MD(gad,A,D,ssAC);
        } else {
          ADn = newAD_MD_adam(gad,A,D,ssAC,iter,mt,vt,control);
        }
      }
      An = Rcpp::as<arma::mat>(ADn["A"]);
      Dn = Rcpp::as<arma::cube>(ADn["D"]);
      Cn = D2C(Dn);
    }
    // arma::vec Mn = newM(mu,R,Z,ssAC);
    arma::mat Ln = newL(mu,L,Z,ssAC,corFLAG);
    arma::mat Zn(arma::size(Z));
    if(ii <= tunelim){
      Zn = Z;
    } else {
      if(sampler == "ULA"){
        Zn = newZ_ULA(Y,PI,Z,A,C,mu,R,spD,ssZ,knots,degree);
      } else if(sampler == "MALA"){
        Zn = newZ_MALA(Y,PI,Z,A,C,mu,R,ar,spD,ssZ,knots,degree,basis);
      } else if(sampler == "RWMH"){
        Zn = newZ_RWMH(Y,PI,Z,A,C,mu,R,ar,ssZ,knots,degree,basis);
      }
    }
    Rcpp::List MRn = newpMR(Zn,pM,pR,iter);
    pMn =  Rcpp::as<arma::mat>(MRn["posM"]);
    pRn =  Rcpp::as<arma::cube>(MRn["posR"]);
    A = An;
    C = Cn;
    D = Dn;
    // mu = Mn;
    L = Ln;
    R = Ln*Ln.t();
    Z = Zn;
    pM = pMn;
    pR = pRn;

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
      double fzll = arma::accu(fz(Z,mu,R));
      lltrace(iter/10-1,0) = fzll;
      double fyzll = arma::accu(fyz(Y,PI));
      lltrace(iter/10-1,1) = fyzll;
      double cdll = fyzll + fzll;
      artrace(iter/10-1) = ar/(n*iter);

      if(verbose){
        if(iter >= burnin){
          if(sampler == "ULA"){
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (estimation phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ")";
          } else {
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (estimation phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ", AR (" << sampler << "): " << std::to_string(ar/(n*iter)) << ")";
          }
        } else {
          if(sampler == "ULA"){
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (burn-in phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ")";
          } else {
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (burn-in phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ", AR (" << sampler << "): " << std::to_string(ar/(n*iter)) << ")";
          }
        }
      }

      if(stopFLAG & (maxvalue < epsStop)) {
        iter++;
        break;
      }
    }

    if(iter >= burnin){
      Aout += A;
      Cout += C;
      Dout += D;
      Mout += mu;
      Rout += R;
      Zout += Z;
      pMout += pM;
      pRout += pR;
    }
    if(ii > tunelim){
      iter++;
    }
  }
  Aout /= (iter - burnin);
  Cout /= (iter - burnin);
  Dout /= (iter - burnin);
  Mout /= (iter - burnin);
  Rout /= (iter - burnin);
  Zout /= (iter - burnin);
  pMout /= (iter - burnin);
  pRout /= (iter - burnin);


  const int nsim = control["nsim"];
  double llk(0), BIC(0), AIC(0);
  if(nsim != 0){
    // llk = fy_gapmCDM(Y,Aout,Cout,Mout,Rout,control);
    llk = fy_gapmCDM_IS(Y,Aout,Cout,Mout,Rout,
                        pMout,pRout,control);
    BIC = -2*llk + std::log(n)*Aout.n_elem;
    AIC = -2*llk + 2*Aout.n_elem;
  }
  arma::mat Uout = Z2U(Zout);
  spObj = SpU_bsp(Uout,knots,degree);
  spM = spObj.slice(0);
  PI = prob(Aout,Cout,spM);

  if(traceFLAG){
    return Rcpp::List::create(Rcpp::Named("A") = Aout,
                              Rcpp::Named("C") = Cout,
                              Rcpp::Named("D") = Dout,
                              Rcpp::Named("mu") = Mout,
                              Rcpp::Named("R") = Rout,
                              Rcpp::Named("Z") = Zout,
                              Rcpp::Named("U") = Uout,
                              Rcpp::Named("PI") = PI,
                              Rcpp::Named("llk") = llk,
                              Rcpp::Named("BIC") = BIC,
                              Rcpp::Named("AIC") = AIC,
                              Rcpp::Named("cdllk.trace") = lltrace,
                              Rcpp::Named("ar.trace") = artrace,
                              Rcpp::Named("theta.trace") = patrace,
                              Rcpp::Named("posMu") = pMout,
                              Rcpp::Named("posR") = pRout);
  } else {
    return Rcpp::List::create(Rcpp::Named("A") = Aout,
                              Rcpp::Named("C") = Cout,
                              Rcpp::Named("D") = Dout,
                              Rcpp::Named("mu") = Mout,
                              Rcpp::Named("R") = Rout,
                              Rcpp::Named("Z") = Zout,
                              Rcpp::Named("U") = Uout,
                              Rcpp::Named("PI") = PI,
                              Rcpp::Named("llk") = llk,
                              Rcpp::Named("BIC") = BIC,
                              Rcpp::Named("AIC") = AIC,
                              Rcpp::Named("posMu") = pMout,
                              Rcpp::Named("posR") = pRout);
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
    if(sampler == "ULA"){
      double ssZ = h * std::pow(ii,-.33) ;
      Zn = newZ_ULA(Ytrain,PI,Z,A,C,mu,R,spD,ssZ,knots,degree);
    } else if(sampler == "MALA"){
      double ssZ = h * std::pow(q,-.33) ;
      Zn = newZ_MALA(Ytrain,PI,Z,A,C,mu,R,ar,spD,ssZ,knots,degree,basis);
    } else if(sampler == "RWMH"){
      double ssZ = h * std::pow(q,-1) ;
      Zn = newZ_RWMH(Ytrain,PI,Z,A,C,mu,R,ar,ssZ,knots,degree,basis);
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
  arma::mat PI = prob_aCDM(G,U,Apat);
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
  const double epsStop = control["stop.eps"];
  const double epsTune = control["tune.eps"];
  const bool verbose = control["verbose"];
  const bool stopFLAG = control["stop.atconv"];
  const bool traceFLAG = control["return.trace"];
  const bool corFLAG = control["cor.R"];
  const std::string sampler = Rcpp::as<std::string>(control["sampler"]);
  double h = control["h"];
  double g = control["gamma"];

  const int n = Y.n_rows;
  const int p = Y.n_cols;
  const int q = R.n_cols;

  arma::mat L = arma::chol(R,"lower");
  arma::mat PI(arma::size(Y));
  arma::mat Zout(arma::size(Z));
  arma::mat Gout(arma::size(G)), Gn(arma::size(G));
  arma::vec Mout(arma::size(mu)), Mn(arma::size(mu));
  arma::mat Rout(arma::size(R));
  // arma::mat posR(q,q);
  arma::cube pR(q,q,n), pRn(q,q,n), pRout(q,q,n);
  cube2eye(pR);
  // arma::rowvec posM(q);
  arma::mat pM(arma::size(Z)), pMn(arma::size(Z)), pMout(arma::size(Z));

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
      ssG = epsTune/n ;
      if(verbose & (ii % 10 == 0)) Rcpp::Rcout << "\r Iteration: " << std::setw(5) << ii << " (tuning phase)";
      ar = 0 ;
      ssZ = h ;
    } else {
      // ssG = g/n * std::pow(iter,-.833) ;
      ssG = g/n * std::pow(iter,-.51) ;
      if(sampler == "ULA"){
        ssZ = h * std::pow(iter,-.333) ; // * std::pow(q,-.333)
      } else if(sampler == "MALA"){
        ssZ = h * std::pow(q,-.333) ; // * std::pow(iter,-.333)
      } else if(sampler == "RWMH"){
        ssZ = h * std::pow(q,-1) ;
      }
    }
    PI = prob_aCDM(G,U,Apat);
    Rcpp::List dG = d1G(Y,U,PI,G,Apat,Qmatrix);
    arma::mat Gn = newG_MD(dG,G,ssG);
    arma::vec Mn = newM(mu,R,Z,ssG);
    arma::mat Ln = newL(mu,L,Z,ssG,corFLAG);
    arma::mat Zn(arma::size(Z));
    if(ii <= tunelim){
      Zn = Z;
    } else {
      if(sampler == "ULA"){
        Zn = newZ_ULA_aCDM(Y,PI,Z,Qmatrix,Apat,G,mu,R,ssZ);
      } else if(sampler == "MALA"){
        Zn = newZ_MALA_aCDM(Y,PI,Z,Qmatrix,Apat,G,mu,R,ssZ,ar);
      } else if(sampler == "RWMH"){
        Zn = newZ_RWMH_aCDM(Y,PI,Z,G,Apat,mu,R,ssZ,ar);
      }
    }
    Rcpp::List MRn = newpMR(Zn,pM,pR,iter);
    pMn =  Rcpp::as<arma::mat>(MRn["posM"]);
    pRn =  Rcpp::as<arma::cube>(MRn["posR"]);
    G = Gn;
    mu = Mn;
    L = Ln;
    R = Ln*Ln.t();
    Z = Zn;
    pM = pMn;
    pR = pRn;

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
      double fzll = arma::accu(fz(Z,mu,R));
      lltrace(iter/10-1,0) = fzll;
      double fyzll = arma::accu(fyz(Y,PI));
      lltrace(iter/10-1,1) = fyzll;
      double cdll = fyzll + fzll;
      artrace(iter/10-1) = ar/(n*iter);

      if(verbose){
        if(iter >= burnin){
          if(sampler == "ULA"){
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (estimation phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ")";
          } else {
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (estimation phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ", AR (" << sampler << "): " << std::to_string(ar/(n*iter)) << ")";
          }
        } else {
          if(sampler == "ULA"){
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (burn-in phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ")";
          } else {
            Rcpp::Rcout << "\r Iteration: " << std::setw(5) << iter << " (burn-in phase, CD-llk: " << std::to_string(cdll) <<
              ", max (abs) change: " << std::to_string(maxvalue) << ", AR (" << sampler << "): " << std::to_string(ar/(n*iter)) << ")";
          }
        }
      }

      if(stopFLAG & (maxvalue < epsStop)) {
        iter++;
        break;
      }
    }

    if(iter >= burnin){
      Gout += G;
      Mout += mu;
      Rout += R;
      Zout += Z;
      pMout += pM;
      pRout += pR;
    }
    if(ii > tunelim){
      iter++;
      // if(iter == iterlim*0.75) itAC = 1;
      // if(iter % 2000 == 0) itG = 1;
    }
  }
  Gout /= (iter - burnin);
  Mout /= (iter - burnin);
  Rout /= (iter - burnin);
  Zout /= (iter - burnin);
  pMout /= (iter - burnin);
  pRout /= (iter - burnin);

  const int nsim = control["nsim"];
  double llk(0), BIC(0), AIC(0);
  if(nsim != 0){
    // llk = fy_aCDM(Y,Gout,Qmatrix,Apat,Mout,Rout,control);
    llk = fy_aCDM_IS(Y,Gout,Qmatrix,Apat,Mout,Rout,
                     pMout,pRout,control);
    BIC = -2*llk + std::log(n)*tp;
    AIC = -2*llk + 2*tp;
  }

  arma::mat Uout = Z2U(Zout);
  PI = prob_aCDM(Gout,Uout,Apat);

  if(traceFLAG){
    return Rcpp::List::create(Rcpp::Named("G") = Gout,
                              Rcpp::Named("mu") = Mout,
                              Rcpp::Named("R") = Rout,
                              Rcpp::Named("Z") = Zout,
                              Rcpp::Named("U") = Uout,
                              Rcpp::Named("PI") = PI,
                              Rcpp::Named("llk") = llk,
                              Rcpp::Named("BIC") = BIC,
                              Rcpp::Named("AIC") = AIC,
                              Rcpp::Named("cdllk.trace") = lltrace,
                              Rcpp::Named("ar.trace") = artrace,
                              Rcpp::Named("theta.trace") = patrace,
                              Rcpp::Named("posMu") = pMout,
                              Rcpp::Named("posR") = pRout);
  } else {
    return Rcpp::List::create(Rcpp::Named("G") = Gout,
                              Rcpp::Named("mu") = Mout,
                              Rcpp::Named("R") = Rout,
                              Rcpp::Named("Z") = Zout,
                              Rcpp::Named("U") = Z2U(Zout),
                              Rcpp::Named("PI") = PI,
                              Rcpp::Named("llk") = llk,
                              Rcpp::Named("BIC") = BIC,
                              Rcpp::Named("AIC") = AIC,
                              Rcpp::Named("posMu") = pMout,
                              Rcpp::Named("posR") = pRout);
  }
}

// [[Rcpp::export]]
Rcpp::List apmCDM_cv_rcpp(arma::mat& Ytrain, arma::mat& Ytest, arma::mat& G,
                          arma::mat& Qmatrix, arma::mat& Apat,
                          arma::vec& mu, arma::mat& R, arma::mat& Z, Rcpp::List& control){

  const int nsim = control["nsim"];
  const double h = control["h"];
  const bool verbose = control["verbose"];
  const std::string sampler = Rcpp::as<std::string>(control["sampler"]);

  const int n = Ytrain.n_rows;
  const int p = Ytrain.n_cols;
  const int q = mu.n_elem;

  arma::mat piH(n,p);
  arma::uvec isNA = arma::find_nan(Ytrain);
  const int nmis = isNA.n_elem;

  for(int ii = 1; ii <= nsim; ii++){
    if (ii % 2 == 0) Rcpp::checkUserInterrupt();
    arma::mat U = Z2U(Z);
    arma::mat PI = prob_aCDM(G,U,Apat);
    arma::mat Zn(arma::size(Z));
    double ar = 0;
    if(sampler == "ULA"){
      double ssZ = h * std::pow(ii,-.33) ;
      Zn = newZ_ULA_aCDM(Ytrain,PI,Z,Qmatrix,Apat,G,mu,R,ssZ);
    } else if(sampler == "MALA"){
      double ssZ = h * std::pow(q,-.33) ;
      Zn = newZ_MALA_aCDM(Ytrain,PI,Z,Qmatrix,Apat,G,mu,R,ssZ,ar);
    } else if(sampler == "RWMH"){
      double ssZ = h * std::pow(q,-1) ;
      Zn = newZ_RWMH_aCDM(Ytrain,PI,Z,G,Apat,mu,R,ssZ,ar);
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
