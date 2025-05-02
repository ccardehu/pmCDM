#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <splines2Armadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(splines2)]]

using namespace Rcpp;
using namespace arma;

arma::mat Z2U(arma::mat& Z){
  const int n = Z.n_rows;
  const int p = Z.n_cols;
  arma::mat out(n,p);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < p; j++){
      out(i,j) = R::pnorm(Z(i,j), 0.0, 1.0, 1, 0);
    }
  }
  return(out);
}

// [[Rcpp::export]]
arma::cube D2C(arma::cube& D){
  const int p = D.n_rows;
  const int tp = D.n_cols;
  const int q = D.n_slices;
  arma::cube out(arma::size(D));
  for(int j = 0; j < q; j++){
    arma::mat Dq = arma::exp(D.slice(j));
    arma::mat tmp(p,tp);
    for(int i = 0; i < p; i++){
      arma::rowvec Dqi = Dq.row(i);
      double denp = arma::accu(Dqi);
      arma::rowvec nump = arma::cumsum(Dqi)/denp;
      tmp.row(i) = nump;
    }
    out.slice(j) = tmp;
  }
  return(out);
}

arma::mat cumsumMat(arma::vec& expd){
  const int tp = expd.n_elem;
  arma::vec outv(tp);
  for(int i = 1; i < tp; i++){
    outv(i-1) = arma::accu(expd(arma::span(i,tp-1)));
  }
  arma::mat out(tp,tp);
  out.each_row() = outv.t();
  return(out);
}

arma::vec ProxD(arma::vec& y){
  // Taken from https://arxiv.org/pdf/1309.1541.pdf
  // See also https://angms.science/doc/CVX/Proj_simplex.pdf
  // and https://arxiv.org/pdf/1101.6081.pdf
  const int m = y.size() ;
  const arma::vec u = arma::sort(y, "descend") ;
  arma::vec rho(m), out1(m), out2(m);
  for(int i = 0; i < m; i++){
    double ii = i+1;
    out1(i) = u(i) + (1/ii)* (1 - arma::accu(u(arma::span(0,i)))) ;
  }
  arma::uvec idx = arma::find(out1 > 0) ;
  arma::uword id = idx.max() ;
  double lambda = out1(id) - u(id) ;
  for(int i = 0; i < m; i++){
    out2(i) = (y(i) + lambda > 0) ? (y(i) + lambda) : 0 ;
  }
  return(out2);
}

arma::mat ProxL(arma::mat& L){
  int q = L.n_rows;
  arma::mat Lout(q,q);
  for(int i = 0; i < q; i++){
    double cross = arma::dot(L.row(i),L.row(i));
    Lout.row(i) = L.row(i)/std::sqrt(cross);
  }
  return(Lout);
}

arma::mat rmvNorm(const int n, arma::vec& mu, arma::mat& R){
  const int q = R.n_cols;
  arma::mat Y(n,q), eigm(q,q);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < q; j++){
      Y(i,j) = R::rnorm(0,1);
    }
  }
  arma::vec eigv(q);
  arma::eig_sym(eigv,eigm,R);
  arma::mat teigm = eigm.t();
  teigm.each_col() %= arma::sqrt(eigv);
  return arma::repmat(mu, 1, n).t() + Y * (eigm * teigm);
}

arma::mat rmvStNorm(const int n, const int q){
  arma::mat Y(n,q) ;
  for(int i = 0; i < n; i++){
    for(int j = 0; j < q; j++){
      Y(i,j) = R::rnorm(0,1);
    }
  }
  return (Y);
}

double dmvNorm(arma::vec& y, arma::vec& mu, arma::mat& R, const bool log = true){
  const int q = y.size() ;
  double lR, sign, out(0.0);
  bool ok = arma::log_det(lR, sign, R);
  if(ok) out = -0.5*(q*std::log(2*M_PI) + lR + arma::as_scalar((y - mu).t()*(arma::inv_sympd(R, arma::inv_opts::allow_approx)*(y - mu)))) ;
  if(log) return(out);
  else return(std::exp(out));
}

arma::mat dmvStNorm(arma::mat& Z){
  const int n = Z.n_rows;
  const int q = Z.n_cols;
  arma::mat out(n,q) ;
  for(int j = 0; j < q; j++){
    for(int m = 0; m < n; m++){
      out(m,j) = R::dnorm(Z(m,j),0.0,1.0,0);
    }
  }
  return (out);
}

// double UiAk(arma::rowvec& Ui, arma::rowvec& Ak){
//   return arma::prod(arma::pow(Ui,Ak) % arma::pow(1.0 - Ui, 1.0 - Ak));
// }

arma::mat UA(arma::mat& U, arma::mat& Apat){
  const int n = U.n_rows;
  const int K = Apat.n_rows;
  arma::mat out(n,K);
  for(int k = 0; k < K; k++){
    arma::mat tAk(arma::size(U));
    tAk.each_row() = Apat.row(k);
    out.col(k) = arma::prod(arma::pow(U,tAk) % arma::pow(1.0 - U, 1.0 - tAk), 1);
  }
  return(out);
}

// arma::rowvec dUiAk(arma::rowvec& Ui, arma::rowvec& Ak){
//   return (Ak/Ui - (1-Ak)/(1-Ui));
// }

arma::mat dUA(arma::mat& U, arma::rowvec& Ak){
  arma::mat tAk(arma::size(U));
  tAk.each_row() = Ak;
  return (tAk/U - (1-tAk)/(1-U));
}

// [[Rcpp::export]]
Rcpp::List genpar(const int p, const int q,
                  const double probSparse,
                  arma::vec& knots, const int degree,
                  const std::string& basis){
  const int np = knots.size();
  arma::mat A(p,q);
  arma::cube C(p,np+degree,q);
  arma::cube D(p,np+degree,q);
  arma::vec sparse(p);
  for(int i = 0; i < p; i++){
    sparse(i) = R::rbinom(1,probSparse);
    if(sparse(i) == 1){
      arma::uvec iA = arma::regspace<uvec>(0,q-1);
      int i2A = arma::conv_to<int>::from(Rcpp::RcppArmadillo::sample(iA,1,false,1/iA.size()));
      A(i,i2A) = 1;
    } else {
      arma::rowvec tA(q+1);
      tA.tail(1) = 1;
      Rcpp::NumericVector tARcpp = Rcpp::runif(q-1);
      tA(span(1,tARcpp.size())) = arma::sort(Rcpp::as<arma::rowvec>(tARcpp));
      tA = arma::diff(tA);
      A.row(i) = tA;
    }
    if(basis == "is"){
      for(int j = 0; j < q; j++){
        arma::rowvec tC(np + degree + 1);
        tC.tail(1) = 1;
        Rcpp::NumericVector tCRcpp = Rcpp::runif(np + degree - 1);
        tC(span(1,tCRcpp.size())) = arma::sort(Rcpp::as<arma::rowvec>(tCRcpp));
        tC = arma::diff(tC);
        C.slice(j).row(i) = tC;
      }
    } else {
      for(int j = 0; j < q; j++){
        Rcpp::NumericVector tDRcpp = Rcpp::rnorm(np + degree - 1L, 0.0, 2.0); // np + degree + 1
        arma::rowvec tD0 = Rcpp::as<arma::rowvec>(tDRcpp); // arma::sort()
        arma::rowvec tD1(np + degree);
        tD1(arma::span(0,np + degree - 2L)) = tD0;
        D.slice(j).row(i) = tD1;
      }
    }
  }
  if(basis != "is") C = D2C(D);
  return Rcpp::List::create(Rcpp::Named("A") = A,
                            Rcpp::Named("C") = C,
                            Rcpp::Named("D") = D);
}

// [[Rcpp::export]]
arma::mat genpar_aCDM(arma::mat& Qmatrix, const double maxG0){
  const int p = Qmatrix.n_rows;
  const int q = Qmatrix.n_cols;
  arma::mat G(p,q+1);
  for(int j = 0; j < p; j++){
    arma::rowvec Qj = Qmatrix.row(j);
    arma::uvec iQj = arma::find(Qj == 1) + 1;
    arma::uvec iGj(iQj.n_elem + 1);
    iGj.subvec(1, iQj.n_elem) = iQj;
    double tmpb0 = R::runif(0.0,maxG0);
    if(iQj.n_elem > 1){
      arma::rowvec tG(iGj.n_elem);
      tG.tail(1) = 1.0 - tmpb0;
      Rcpp::NumericVector tGRcpp = Rcpp::runif(iQj.n_elem-1,0.0,1.0 - tmpb0);
      tG(arma::span(1,tGRcpp.size())) = arma::sort(Rcpp::as<arma::rowvec>(tGRcpp));
      tG = arma::diff(tG);
      arma::rowvec t2G(tG.n_elem + 1);
      t2G(0) = tmpb0;
      t2G.subvec(1, tG.n_elem) = tG;
      arma::rowvec tGj(q+1);
      tGj(iGj) = t2G;
      G.row(j) = tGj;
    } else {
      arma::rowvec tG(iGj.n_elem);
      tG(0) = tmpb0;
      tG(1) = 1.0 - tmpb0;
      arma::rowvec tGj(q+1);
      tGj(iGj) = tG;
      G.row(j) = tGj;
    }
  }
  return(G);
}

arma::cube SpU_isp(arma::mat& U, arma::vec& knots, const unsigned int deg){
  const arma::vec bouK = {0.0,1.0};
  const int n = U.n_rows;
  const int q = U.n_cols;
  const int np = knots.size() + deg ;
  arma::mat out1(n,q*np);
  arma::mat out2(n,q*np);
  arma::cube out(n,q*np,2);
  for(int j = 0; j < q; j++){
    splines2::ISpline iSp(U.col(j),knots,deg,bouK);
    out1.cols(j*np,(j+1)*np-1) = iSp.basis(false);
    out2.cols(j*np,(j+1)*np-1) = iSp.derivative(1,false);
  }
  out.slice(0) = out1;
  out.slice(1) = out2;
  return(out)  ;
}

arma::cube SpU_bsp(arma::mat& U, arma::vec& knots, const unsigned int deg){
  const arma::vec bouK = {0.0,1.0};
  const int n = U.n_rows;
  const int q = U.n_cols;
  const int np = knots.size() + deg ;
  arma::mat out1(n,q*np);
  arma::mat out2(n,q*np);
  arma::cube out(n,q*np,2);
  for(int j = 0; j < q; j++){
    splines2::BSpline bSp(U.col(j),knots,deg,bouK);
    out1.cols(j*np,(j+1)*np-1) = bSp.basis(false);
    out2.cols(j*np,(j+1)*np-1) = bSp.derivative(1,false);
  }
  out.slice(0) = out1;
  out.slice(1) = out2;
  return(out)  ;
}

arma::mat prob(arma::mat& A, arma::cube& C, arma::mat& ism){
  const int n = ism.n_rows;
  const int p = A.n_rows;
  const int q = C.n_slices;
  const int np = C.n_cols;
  arma::mat out(n,p, arma::fill::zeros);
  for(int i = 0; i < p; i++){
    for(int j = 0; j < q; j++){
      out.col(i) += A(i,j) * ism.cols(j*np,(j+1)*np-1)*(C.slice(j).row(i)).t();
    }
  }
  return(out);
}

arma::mat fyz(arma::mat& Y, arma::mat& PI){
  const int n = Y.n_rows;
  const int p = Y.n_cols;
  arma::mat out(n,p,arma::fill::zeros);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < p; j++){
      if(std::isnan(Y(i,j))) continue;
      out(i,j) = (Y(i,j) > 0) ? std::max(arma::datum::log_min, std::log(PI(i,j))) : std::max(arma::datum::log_min, std::log(1-PI(i,j))) ;
    }
  }
  return(out);
}

arma::mat fz(arma::mat& Z, arma::vec& mu, arma::mat& R){
  const int n = Z.n_rows;
  arma::vec out(n);
  for(int i = 0; i < n; i++){
    arma::vec zrow = Z.row(i).t();
    out(i) = std::max(arma::datum::log_min, dmvNorm(zrow, mu, R, true)) ;
  }
  return(out);
}

// [[Rcpp::export]]
double fy_gapmCDM(arma::mat& Y, arma::mat& A, arma::cube& C,
          arma::vec& mu, arma::mat& R, Rcpp::List& control){

  const unsigned int degree = control["degree"];
  const int nsim = control["nsim"];
  arma::vec knots = control["knots"];
  const bool verbose = control["verbose"];
  const std::string basis = Rcpp::as<std::string>(control["basis"]);

  const int n = Y.n_rows;
  if(verbose) Rcpp::Rcout << "\n Calculating marginal log-likelihood ... ";
  double mllk = 0;
  arma::mat Eobj(n,nsim);
  arma::mat Zsim(n,R.n_cols);
  arma::mat isMo(n,R.n_cols * C.n_cols);
  arma::cube spObj(n,R.n_cols * C.n_cols,2);
  for(int ii = 0; ii < nsim; ii++){
    if (ii % 2 == 0) Rcpp::checkUserInterrupt();
    Zsim = rmvNorm(n,mu,R);
    arma::mat Usim = Z2U(Zsim);
    if(basis == "is"){
      spObj = SpU_isp(Usim,knots,degree);
      isMo = spObj.slice(0);
    } else {
      spObj = SpU_bsp(Usim,knots,degree);
      isMo = spObj.slice(0);
    }
    arma::mat piH = prob(A,C,isMo);
    Eobj.col(ii) = arma::sum(fyz(Y,piH),1);
  }
  arma::vec maxEobj(n);
  for(int m = 0; m < n; m++){
    maxEobj(m) = arma::max(Eobj.row(m));
  }
  Eobj.each_col() -= maxEobj;
  arma::mat EEobj = arma::exp(Eobj);
  arma::vec V1 = arma::log(arma::sum(EEobj,1));
  mllk = arma::accu(V1) + arma::accu(maxEobj) - n*std::log(nsim);
  if(verbose) Rcpp::Rcout << "\r Calculating marginal log-likelihood ... (m-llk: " << std::to_string(mllk) << ")";
  return(mllk);
}

// [[Rcpp::export]]
double fy_gapmCDM_IS(arma::mat& Y, arma::mat& A, arma::cube& C,
                     arma::vec& mu, arma::mat& R,
                     arma::rowvec& pmur, arma::mat& pR, Rcpp::List& control){

  const unsigned int degree = control["degree"];
  const int nsim = control["nsim"];
  arma::vec knots = control["knots"];
  const bool verbose = control["verbose"];
  const std::string sampler = Rcpp::as<std::string>(control["sampler"]);
  const std::string basis = Rcpp::as<std::string>(control["basis"]);

  const int n = Y.n_rows;
  if(verbose) Rcpp::Rcout << "\n Calculating marginal log-likelihood (via IS) ... ";
  double mllk = 0;
  arma::mat Eobj(n,nsim);
  arma::mat Zsim(n,R.n_cols);
  arma::mat isMo(n,R.n_cols * C.n_cols);
  arma::mat isDo(n,R.n_cols * C.n_cols);
  arma::cube spObj(n,R.n_cols * C.n_cols,2);
  arma::vec pmu = pmur.t();
  arma::mat fixpR(arma::size(R), arma::fill::eye);
  arma::mat pr = 1.5*fixpR + pR;
  for(int ii = 0; ii < nsim; ii++){
    if (ii % 2 == 0) Rcpp::checkUserInterrupt();
    Zsim = rmvNorm(n,pmu,pR);
    arma::mat Usim = Z2U(Zsim);
    if(basis == "is"){
      spObj = SpU_isp(Usim,knots,degree);
      isMo = spObj.slice(0);
    } else {
      spObj = SpU_bsp(Usim,knots,degree);
      isMo = spObj.slice(0);
    }
    arma::mat piH = prob(A,C,isMo);
    Eobj.col(ii) = arma::sum(fyz(Y,piH),1) + fz(Zsim,mu,R) - fz(Zsim,pmu,pr);
  }
  arma::vec maxEobj(n);
  for(int m = 0; m < n; m++){
    maxEobj(m) = arma::max(Eobj.row(m));
  }
  Eobj.each_col() -= maxEobj;
  arma::mat EEobj = arma::exp(Eobj);
  arma::vec V1 = arma::log(arma::sum(EEobj,1));
  mllk = arma::accu(V1) + arma::accu(maxEobj) - n*std::log(nsim);
  if(verbose) Rcpp::Rcout << "\r Calculating marginal log-likelihood (via IS) ... (m-llk: " << std::to_string(mllk) << ")";
  return(mllk);
}

Rcpp::List d1AC(arma::mat& Y, arma::mat& PI, arma::mat& ism,
                arma::mat& A, arma::cube& C){
  arma::mat YmPI = (Y - PI)/(PI % (1-PI));
  // arma::mat EmPI = (-(arma::pow(PI,2) - 2.0*(PI%Y) + Y)/(arma::pow(PI-1,2)%arma::pow(PI,2)));
  const int p = YmPI.n_cols;
  const int q = C.n_slices;
  const int np = C.n_cols;
  const int tp = A.size() + C.size();
  arma::uvec ia = arma::regspace<arma::uvec>(0, A.size());
  arma::umat iA(ia.begin(), A.n_rows, A.n_cols, false);
  arma::uvec ic = arma::regspace<arma::uvec>(A.size(), C.size() + A.size());
  arma::ucube iC(ic.begin(), C.n_rows, C.n_cols, C.n_slices, false);
  arma::vec out(tp);
  // arma::mat Asq = arma::pow(A,2);
  // arma::mat out2(tp,tp);
  for(int i = 0; i < p; i++){
    arma::vec YmPIcol = YmPI.col(i);
    // arma::vec EmPIcol = EmPI.col(i);
    arma::uvec noNA = arma::find_finite(YmPIcol);
    for(int j = 0; j < q; j++){
      arma::uvec jcols = arma::regspace<uvec>(j*np,(j+1)*np-1);
      arma::uword idA = iA(i,j);
      arma::uvec idC = iC.slice(j).row(i).t();
      out(idA) = arma::as_scalar(YmPIcol.elem(noNA).t() * (ism(noNA,jcols)*(C.slice(j).row(i)).t()));
      // out2(idA,idA) = arma::as_scalar((ism(noNA,jcols)*(C.slice(j).row(i)).t()).t()*arma::diagmat(EmPIcol.elem(noNA))*(ism(noNA,jcols)*(C.slice(j).row(i)).t()));
      arma::vec A2 = ism(noNA,jcols).t() * YmPIcol.elem(noNA);
      // arma::mat A3 = ism(noNA,jcols).t() * arma::diagmat(EmPIcol.elem(noNA)) * ism(noNA,jcols);
      out(idC) = A(i,j) * A2 ; //A2.t()
      // out2(idC,idC) = Asq(i,j) * arma::diagmat(A3) ;
    }
  }
  // out2 = -1.0/arma::diagmat(out2);
  return Rcpp::List::create(Rcpp::Named("grad") = out,
                            // Rcpp::Named("hess") = out2,
                            Rcpp::Named("iA") = iA,
                            Rcpp::Named("iC") = iC);
}

// Rcpp::List d2AC(arma::mat& YmPI, arma::mat& ism,
//                 arma::mat& A, arma::cube& C){
//   const int p = YmPI.n_cols;
//   const int q = C.n_slices;
//   const int np = C.n_cols;
//   const int tp = A.size() + C.size();
//   arma::uvec ia = arma::regspace<arma::uvec>(0, A.size());
//   arma::umat iA(ia.begin(), A.n_rows, A.n_cols, false);
//   arma::uvec ic = arma::regspace<arma::uvec>(A.size(), C.size() + A.size());
//   arma::ucube iC(ic.begin(), C.n_rows, C.n_cols, C.n_slices, false);
//   arma::mat Asq = arma::pow(A,2);
//   arma::mat out(tp,tp);
//   for(int i = 0; i < p; i++){
//     for(int j = 0; j < q; j++){
//       arma::uword idA = iA(i,j);
//       arma::uvec idC = iC.slice(j).row(i).t();
//       out(idA,idA) = arma::as_scalar((ism.cols(j*np,(j+1)*np-1)*(C.slice(j).row(i)).t()).t()*arma::diagmat(YmPI.col(i))*(ism.cols(j*np,(j+1)*np-1)*(C.slice(j).row(i)).t()));
//       arma::mat A2 = ism.cols(j*np,(j+1)*np-1).t() * arma::diagmat(YmPI.col(i)) * ism.cols(j*np,(j+1)*np-1);
//       out(idC,idC) =  Asq(i,j) * A2 ;
//     }
//   }
//   return Rcpp::List::create(Rcpp::Named("hess") = out,
//                             Rcpp::Named("iA") = iA,
//                             Rcpp::Named("iC") = iC);
// }

arma::mat d1CdD(arma::vec& d){
  const int tp = d.n_elem;
  arma::vec expd = arma::exp(d);
  double den = std::pow(arma::accu(expd),2);
  arma::vec num = arma::cumsum(expd);
  arma::mat csM = cumsumMat(expd);
  arma::mat nuM(tp,tp), out(tp,tp);
  nuM.each_row() = num.t();
  out.each_col() = expd;
  arma::vec zeroC(tp);
  out.tail_cols(1) = zeroC;

  arma::uvec itl = arma::trimatl_ind(arma::size(out),-1);
  arma::uvec itu = arma::trimatu_ind(arma::size(out));

  out(itl) %= -nuM(itl);
  out(itu) %= csM(itu);
  out /= den ;
  return(out);
}

arma::mat d2CdD(arma::vec& d, arma::mat& dCdD){
  arma::vec expd = arma::exp(d);
  double den = arma::accu(expd);
  arma::vec num = 2*expd - den;
  arma::vec dCdDvec = arma::diagvec(dCdD);
  arma::vec out = num/den % dCdDvec;
  arma::mat outM = -1.0*(arma::diagmat(out)) ;
  return(outM);
}

Rcpp::List dCdD(arma::vec& d){
  const int tp = d.n_elem;
  arma::vec expd = arma::exp(d);
  arma::mat d1(tp,tp);
  arma::vec N = arma::cumsum(expd);
  double D = arma::accu(expd);

  // For d1CdD
  arma::mat expDM(tp,tp);
  expDM.each_col() = expd;
  arma::mat Nm(tp,tp);
  Nm.each_row() = N.t();
  arma::mat DmN = D - Nm;
  Nm %= expDM;
  DmN %= expDM;
  arma::uvec iLm = arma::trimatl_ind(arma::size(d1),-1);
  arma::uvec iUm = arma::trimatu_ind(arma::size(d1));
  d1(iUm) = DmN(iUm);
  d1(iLm) = -Nm(iLm);

  // // For d2CdD
  // arma::mat NmDmN = Nm - DmN;
  // NmDmN %= expDM;
  // arma::cube d2(tp,tp,tp);
  // arma::mat tM2(tp,tp,arma::fill::value(22));
  // tM2(iLm).fill(11);
  //
  // for(int i = 0; i < tp; i++){
  //   arma::mat tM1(tp,i,arma::fill::value(22));
  //   arma::uvec tiUm = arma::trimatu_ind(arma::size(tM1));
  //   tM1(tiUm).fill(11);
  //   arma::mat tM2a = tM2.cols(i,tp-1);
  //   arma::mat tmat = arma::join_rows(tM1,tM2a);
  //   tmat.shed_row(i);
  //   arma::rowvec ri = -d1.row(i)*(expd(i)-(D-expd(i)));
  //   tmat.insert_rows(i,ri);
  //   arma::uvec i11 = arma::find(tmat == 11);
  //   arma::uvec i22 = arma::find(tmat == 22);
  //   tmat(i22) = -2.0*d1(i22)*expd(i);
  //   tmat(i11) = NmDmN(i11) * expd(i);
  //   d2.slice(i) = tmat;
  //   d2.slice(i).tail_cols(1).fill(0.0);
  // }
  return Rcpp::List::create(Rcpp::Named("d1CdD") = d1/std::pow(D,2));//,
                            // Rcpp::Named("d2CdD") = d2/std::pow(D,3));
}

Rcpp::List d1AD(arma::mat& Y, arma::mat& PI, arma::mat& ism,
                arma::mat& A, arma::cube& D){
  arma::mat YmPI = (Y - PI)/(PI % (1-PI));
  // arma::mat YmPI2 = (-(arma::pow(PI,2) - 2.0*(PI%Y) + Y)/(arma::pow(PI-1,2)%arma::pow(PI,2)));
  const int p = YmPI.n_cols;
  const int q = D.n_slices;
  const int np = D.n_cols;
  const int tp = A.size() + D.size();
  arma::cube C = D2C(D);
  arma::uvec ia = arma::regspace<arma::uvec>(0, A.size());
  arma::umat iA(ia.begin(), A.n_rows, A.n_cols, false);
  arma::uvec ic = arma::regspace<arma::uvec>(A.size(), C.size() + A.size());
  arma::ucube iC(ic.begin(), C.n_rows, C.n_cols, C.n_slices, false);
  arma::vec out(tp);
  // arma::mat Asq = arma::pow(A,2);
  // arma::mat out2(tp,tp);
  for(int i = 0; i < p; i++){
    arma::vec YmPIcol = YmPI.col(i);
    // arma::vec YmPI2col = YmPI2.col(i);
    arma::uvec noNA = arma::find_finite(YmPIcol);
    for(int j = 0; j < q; j++){
      arma::uvec jcols = arma::regspace<uvec>(j*np,(j+1)*np-1);
      arma::uword idA = iA(i,j);
      arma::uvec idC = iC.slice(j).row(i).t();
      out(idA) = arma::as_scalar(YmPIcol.elem(noNA).t() * (ism(noNA,jcols)*(C.slice(j).row(i)).t()));
      arma::vec Dij = D.slice(j).row(i).t();
      Rcpp::List dD = dCdD(Dij);
      arma::mat dD1 = dD["d1CdD"];
      // arma::cube dD2 = dD["d2CdD"];
      // arma::mat M1(arma::size(dD1));
      arma::vec V1 = ism(noNA,jcols).t() * YmPIcol.elem(noNA);
      arma::vec V2 = (dD1 * V1);
      // for(int i = 0; i < np; i++){
      //   M1 += dD2.col_as_mat(i)*V1(i);
      // }
      // arma::mat M2 = (dD1 * ism(noNA,jcols).t()) * arma::diagmat(YmPI2col.elem(noNA)) * (ism(noNA,jcols) * dD1.t());
      out(idC) = A(i,j) * V2 ;
      // out2(idC,idC) = Asq(i,j) * M2 +  A(i,j) * M1;
    }
  }
  return Rcpp::List::create(Rcpp::Named("gs") = out,
                            // Rcpp::Named("hs") = out2,
                            Rcpp::Named("iA") = iA,
                            Rcpp::Named("iD") = iC);
}

arma::cube d1PIdZ(arma::mat& A, arma::cube& C, arma::mat& isd, arma::mat& Z){
  const int p = A.n_rows;
  const int q = C.n_slices;
  const int n = isd.n_rows;
  const int np = C.n_cols;
  arma::mat dZ = dmvStNorm(Z);
  arma::cube out(n,q,p);
  for(int i = 0; i < p; i++){
    for(int j = 0; j < q; j++){
      out.slice(i).col(j) = A(i,j) * (isd.cols(j*np,(j+1)*np-1) * C.slice(j).row(i).t()) % dZ.col(j) ;
    }
  }
  return(out);
}

arma::cube d1PIdZ_aCDM(arma::mat& G, arma::mat& Qmatrix, arma::mat& Z, arma::mat& Apat){
  const int p = G.n_rows;
  const int q = Z.n_cols;
  const int n = Z.n_rows;
  const int K = Apat.n_rows;
  arma::mat dZ = dmvStNorm(Z);
  arma::mat U = Z2U(Z);
  arma::mat UpA = UA(U,Apat);
  arma::vec c1(K, arma::fill::value(1.0));
  arma::mat Apat1 = arma::join_rows(c1,Apat);
  arma::cube out(n,q,p);
  for(int i = 0; i < p; i++){
    arma::rowvec Gi = G.row(i);
    for(int k = 0; k < K; k++){
      arma::rowvec Ak = Apat.row(k);
      arma::mat dUAt = dUA(U,Ak);
      arma::mat tUpA(size(U));
      tUpA.each_col() = UpA.col(k);
      double PIAj = arma::as_scalar(Apat1.row(k) * Gi.t());
      out.slice(i) += PIAj * tUpA % dUAt % dZ;
    }
  }
  return(out);
}

arma::mat d1PostZ(arma::mat& YmPI, arma::mat& Z, arma::mat& isd,
                  arma::mat& A, arma::cube& C, arma::vec& mu, arma::mat& R){
  const int n = YmPI.n_rows;
  const int p = YmPI.n_cols;
  const int q = C.n_slices;
  arma::mat out(n,q);
  arma::mat iR = arma::inv_sympd(R, arma::inv_opts::allow_approx);
  arma::cube dpidz = d1PIdZ(A,C,isd,Z);
  for(int i = 0; i < p; i++){
    arma::mat temp = dpidz.slice(i).each_col() % YmPI.col(i);
    for(int j = 0; j < q; j++){
      for(int m = 0; m < n; m++){
        if(std::isnan(temp(m,j))) continue;
        out(m,j) += temp(m,j);
      }
    }
  }
  arma::mat Zcent(Z);
  Zcent.each_row() -= mu.t();
  out -= Zcent*iR;
  return(out);
}

// Rcpp::List aCDM(arma::mat& G, arma::mat& Qmatrix, arma::mat& Z, arma::mat& Apat){
//   const int n = Z.n_rows;
//   const int p = G.n_rows;
//   const int q = Z.n_cols;
//   const int tp = G.n_cols;
//   const int K = Apat.n_rows;
//   arma::mat dZ = dmvStNorm(Z);
//   arma::mat U = Z2U(Z);
//   arma::vec c1(K, arma::fill::value(1.0));
//   arma::mat Apat1 = arma::join_rows(c1,Apat);
//   arma::mat out1(n,p, arma::fill::zeros);
//   arma::cube out2(p,tp,n,arma::fill::zeros);
//   arma::cube out3(n,q,p, arma::fill::zeros);
//   arma::rowvec IAkQj(tp);
//   IAkQj(0) = 1;
//   for(int j = 0; j < p; j++){
//     arma::rowvec Gj = G.row(j);
//     arma::rowvec Qj = Qmatrix.row(j);
//     for(int i = 0; i < n; i++){
//       arma::rowvec Ui = U.row(i);
//       arma::rowvec dZi = dZ.row(i);
//       for(int k = 0; k < K; k++){
//         arma::rowvec Ak = Apat.row(k);
//         arma::rowvec tAkQj = Ak % Qj;
//         double pUiAk = UiAk(Ui,Ak);
//         double PIAj = arma::as_scalar(Apat1.row(k) * Gj.t());
//         out1(i,j) +=  PIAj * pUiAk;
//         IAkQj.subvec(1, arma::size(tAkQj)) = tAkQj;
//         out2.slice(i).row(j) += (IAkQj) * pUiAk;
//         out3.slice(j).row(i) += PIAj * pUiAk * dUiAk(Ui,Ak) % dZi;
//       }
//     }
//   }
//   return Rcpp::List::create(Rcpp::Named("PI") = out1,
//                             Rcpp::Named("d1G") = out2,
//                             Rcpp::Named("d1PIdZ") = out3,
//                             Rcpp::Named("Qmatrix") = Qmatrix) ;
//
// }

arma::mat prob_aCDM(arma::mat& G, arma::mat& U, arma::mat& Apat){
  const int n = U.n_rows;
  const int p = G.n_rows;
  const int K = Apat.n_rows;
  arma::mat UpA = UA(U,Apat);
  arma::vec c1(K, arma::fill::value(1.0));
  arma::mat Apat1 = arma::join_rows(c1,Apat);
  arma::mat out(n,p, arma::fill::zeros);
  for(int i = 0; i < p; i++){
    arma::rowvec Gi = G.row(i);
    for(int k = 0; k < K; k++){
      double PIAj = arma::as_scalar(Apat1.row(k) * Gi.t());
      out.col(i) += PIAj * UpA.col(k);
    }
  }
  return(out);
}


// [[Rcpp::export]]
double fy_aCDM(arma::mat& Y, arma::mat& G, arma::mat& Qmatrix, arma::mat& Apat,
               arma::vec& mu, arma::mat& R, Rcpp::List& control){

  const int nsim = control["nsim"];
  const bool verbose = control["verbose"];

  const int n = Y.n_rows;
  if(verbose) Rcpp::Rcout << "\n Calculating marginal log-likelihood ... ";
  double mllk = 0;
  arma::mat Eobj(n,nsim);
  arma::mat Zsim(n,R.n_cols);
  for(int ii = 0; ii < nsim; ii++){
    if (ii % 2 == 0) Rcpp::checkUserInterrupt();
    Zsim = rmvNorm(n,mu,R);
    arma::mat Usim = Z2U(Zsim);
    // Rcpp::List aCDMlist = aCDM(G,Qmatrix,Zsim,Apat);
    // arma::mat piH = aCDMlist["PI"];
    arma::mat piH = prob_aCDM(G,Usim,Apat);
    Eobj.col(ii) = arma::sum(fyz(Y,piH),1);
  }
  arma::vec maxEobj(n);
  for(int m = 0; m < n; m++){
    maxEobj(m) = arma::max(Eobj.row(m));
  }
  Eobj.each_col() -= maxEobj;
  arma::mat EEobj = arma::exp(Eobj);
  arma::vec V1 = arma::log(arma::sum(EEobj,1));
  mllk = arma::accu(V1) + arma::accu(maxEobj) - n*std::log(nsim);
  if(verbose) Rcpp::Rcout << "\r Calculating marginal log-likelihood ... (m-llk: " << std::to_string(mllk) << ")";
  return(mllk);
}

// [[Rcpp::export]]
double fy_aCDM_IS(arma::mat& Y, arma::mat& G, arma::mat& Qmatrix, arma::mat& Apat,
                  arma::vec& mu, arma::mat& R,
                  arma::rowvec& pmur, arma::mat& pR, Rcpp::List& control){

  const int nsim = control["nsim"];
  const bool verbose = control["verbose"];

  const int n = Y.n_rows;
  if(verbose) Rcpp::Rcout << "\n Calculating marginal log-likelihood (via IS) ... ";
  double mllk = 0;
  arma::mat Eobj(n,nsim);
  arma::mat Zsim(n,R.n_cols);
  arma::vec pmu = pmur.t();
  arma::mat fixpR(arma::size(R), arma::fill::eye);
  arma::mat pr = 1.5*fixpR + pR;
  for(int ii = 0; ii < nsim; ii++){
    if (ii % 2 == 0) Rcpp::checkUserInterrupt();
    Zsim = rmvNorm(n,pmu,pR);
    arma::mat Usim = Z2U(Zsim);
    // Rcpp::List aCDMlist = aCDM(G,Qmatrix,Zsim,Apat);
    // arma::mat piH = aCDMlist["PI"];
    arma::mat piH = prob_aCDM(G,Usim,Apat);
    Eobj.col(ii) = arma::sum(fyz(Y,piH),1) + fz(Zsim,mu,R) - fz(Zsim,pmu,pr);
  }
  arma::vec maxEobj(n);
  for(int m = 0; m < n; m++){
    maxEobj(m) = arma::max(Eobj.row(m));
  }
  Eobj.each_col() -= maxEobj;
  arma::mat EEobj = arma::exp(Eobj);
  arma::vec V1 = arma::log(arma::sum(EEobj,1));
  mllk = arma::accu(V1) + arma::accu(maxEobj) - n*std::log(nsim);
  if(verbose) Rcpp::Rcout << "\r Calculating marginal log-likelihood (via IS) ... (m-llk: " << std::to_string(mllk) << ")";
  return(mllk);
}

// Rcpp::List d1G(arma::mat& Y, Rcpp::List aCDMlist){
//   arma::mat PI = aCDMlist["PI"];
//   arma::mat Qmatrix = aCDMlist["Qmatrix"];
//   arma::cube d1G = aCDMlist["d1G"];
//   arma::mat YmPI = (Y - PI)/(PI % (1-PI));
//   const int p = Y.n_cols;
//   const int q = d1G.n_cols;
//   arma::mat tout(p,q,arma::fill::zeros);
//   arma::vec c1(p, arma::fill::value(1.0));
//   arma::mat Qmatrix1 = arma::join_rows(c1,Qmatrix);
//   for(int j = 0; j < q; j++){
//     for(int i = 0; i < p; i++){
//       arma::vec YmPIcol = YmPI.col(i);
//       arma::uvec noNA = arma::find_finite(YmPIcol);
//       if(Qmatrix1(i,j) != 0){
//         arma::vec d1Gn = d1G.tube(i,j);
//         tout(i,j) = arma::as_scalar(YmPIcol.elem(noNA).t() * d1Gn.elem(noNA));
//       }
//     }
//   }
//   arma::vec out = arma::vectorise(tout);
//   arma::uvec ig = arma::regspace<arma::uvec>(0, tout.size());
//   arma::umat iG(ig.begin(), tout.n_rows, tout.n_cols, false);
//   return Rcpp::List::create(Rcpp::Named("grad") = out,
//                             Rcpp::Named("iG") = iG);
// }

Rcpp::List d1G(arma::mat& Y, arma::mat& U, arma::mat& PI, arma::mat& G,
               arma::mat& Apat, arma::mat& Qmatrix){
  arma::mat YmPI = (Y - PI)/(PI % (1-PI));
  // arma::mat YmPI2 = (-(arma::pow(PI,2) - 2.0*(PI%Y) + Y)/(arma::pow(PI-1,2)%arma::pow(PI,2)));
  const int p = Y.n_cols;
  const int tp = G.n_cols;
  const int K = Apat.n_rows;
  arma::mat UpA = UA(U,Apat);
  arma::mat tout(arma::size(G));
  arma::rowvec IAkQi(tp);
  IAkQi(0) = 1;
  for(int i = 0; i < p; i++){
    arma::rowvec Gi = G.row(i);
    arma::rowvec Qi = Qmatrix.row(i);
    arma::vec YmPIcol = YmPI.col(i);
    arma::uvec noNA = arma::find_finite(YmPIcol);
    for(int k = 0; k < K; k++){
      arma::rowvec Ak = Apat.row(k);
      arma::vec UpAk = UpA.col(k);
      arma::rowvec tAkQi = Ak % Qi;
      IAkQi.subvec(1, arma::size(tAkQi)) = tAkQi;
      double trick = arma::as_scalar(YmPIcol.elem(noNA).t() * UpAk.elem(noNA));
      tout.row(i) += trick * IAkQi;
    }
  }
  arma::vec out = arma::vectorise(tout);
  arma::uvec ig = arma::regspace<arma::uvec>(0, tout.size());
  arma::umat iG(ig.begin(), tout.n_rows, tout.n_cols, false);
  return Rcpp::List::create(Rcpp::Named("grad") = out,
                            Rcpp::Named("iG") = iG);
}

// arma::mat d1PostZ_aCDM(arma::mat& Y, arma::mat& Z, Rcpp::List aCDMlist, arma::vec& mu, arma::mat& R){
//   arma::mat PI = aCDMlist["PI"];
//   arma::mat YmPI = (Y - PI)/(PI % (1-PI));
//   const int n = YmPI.n_rows;
//   const int p = YmPI.n_cols;
//   const int q = mu.n_elem;
//   arma::mat out(n,q);
//   arma::mat iR = arma::inv_sympd(R, arma::inv_opts::allow_approx);
//   arma::cube dpidz = aCDMlist["d1PIdZ"];
//   for(int i = 0; i < p; i++){
//     arma::mat temp = dpidz.slice(i).each_col() % YmPI.col(i);
//     for(int j = 0; j < q; j++){
//       for(int m = 0; m < n; m++){
//         if(std::isnan(temp(m,j))) continue;
//         out(m,j) += temp(m,j);
//       }
//     }
//   }
//   arma::mat Zcent(Z);
//   Zcent.each_row() -= mu.t();
//   out -= Zcent*iR;
//   return(out);
// }

arma::mat d1PostZ_aCDM(arma::mat& Y, arma::mat& PI, arma::mat& Z, arma::mat& Qmatrix, arma::mat& Apat,
                           arma::mat& G, arma::vec& mu, arma::mat& R){
  arma::mat YmPI = (Y - PI)/(PI % (1-PI));
  const int p = Y.n_cols;
  const int n = Y.n_rows;
  const int q = mu.n_elem;
  arma::mat U = Z2U(Z);
  arma::mat UpA = UA(U,Apat);
  arma::mat out(n,q);
  arma::mat iR = arma::inv_sympd(R, arma::inv_opts::allow_approx);
  arma::cube dpidz = d1PIdZ_aCDM(G,Qmatrix,Z,Apat);
  for(int i = 0; i < p; i++){
    arma::mat temp = dpidz.slice(i).each_col() % YmPI.col(i);
    for(int j = 0; j < q; j++){
      for(int m = 0; m < n; m++){
        if(std::isnan(temp(m,j))) continue;
        out(m,j) += temp(m,j);
      }
    }
  }
  arma::mat Zcent(Z);
  Zcent.each_row() -= mu.t();
  out -= Zcent*iR;
  return(out);
}
