#include <RcppArmadillo.h>
#include "utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

Rcpp::List newAC_PGD(Rcpp::List& d1AC, arma::mat& Aold, arma::cube& Cold, double ss){
   arma::vec d1ac = d1AC["grad"];
   // arma::mat d2ac = d1AC["hess"];
   arma::umat iA = d1AC["iA"];
   arma::ucube iC = d1AC["iC"];
   arma::mat Anew(arma::size(Aold));
   arma::cube Cnew(arma::size(Cold));
   const int p = Aold.n_rows;
   const int q = Cold.n_slices;
   for(int i = 0; i < p; i ++){
      arma::uvec idA = iA.row(i).t();
      // arma::vec tA = Aold.row(i).t() + ss*d2ac(idA,idA)*d1ac(idA);
      arma::vec tA = Aold.row(i).t() + ss*d1ac(idA);
      Anew.row(i) = ProxD(tA).t();
      for(int j = 0; j < q; j++){
         if(Aold(i,j) > arma::datum::eps){
            arma::uvec idC = iC.slice(j).row(i).t();
            // arma::vec tC = Cold.slice(j).row(i).t() + ss*d2ac(idC,idC)*d1ac(idC);
            arma::vec tC = Cold.slice(j).row(i).t() + ss*d1ac(idC);
            Cnew.slice(j).row(i) = ProxD(tC).t();
         } else {
            Cnew.slice(j).row(i) = Cold.slice(j).row(i);
         }
      }
   }
   return Rcpp::List::create(Rcpp::Named("A") = Anew,
                             Rcpp::Named("C") = Cnew);
}

Rcpp::List newAD_PGD(Rcpp::List& d1AD,arma::mat& Aold, arma::cube& Dold, double ss){
   arma::vec d1ad = d1AD["grad"];
   // arma::mat d2ad = d1AC["hess"];
   arma::umat iA = d1AD["iA"];
   arma::ucube iD = d1AD["iD"];
   arma::mat Anew(arma::size(Aold));
   arma::cube Dnew(arma::size(Dold));
   const int p = Aold.n_rows;
   const int q = Dold.n_slices;
   for(int i = 0; i < p; i ++){
      arma::uvec idA = iA.row(i).t();
      // arma::vec tA = Aold.row(i).t() + ss*d2ad(idA,idA)*d1ad(idA);
      arma::vec tA = Aold.row(i).t() + ss*d1ad(idA);
      Anew.row(i) = ProxD(tA).t();
      for(int j = 0; j < q; j++){
         if(Aold(i,j) > arma::datum::eps){
            arma::uvec idD = iD.slice(j).row(i).t();
            // arma::vec tD = Dold.slice(j).row(i).t() + ss*d2ad(idD,idD)*d1ad(idD);
            arma::vec tD = Dold.slice(j).row(i).t() + ss*d1ad(idD);
            Dnew.slice(j).row(i) = tD.t();
         } else {
            Dnew.slice(j).row(i) = Dold.slice(j).row(i);
         }
      }
   }
   return Rcpp::List::create(Rcpp::Named("A") = Anew,
                             Rcpp::Named("D") = Dnew);
}

Rcpp::List newAC_MD(Rcpp::List& d1AC,arma::mat& Aold, arma::cube& Cold, double ss){
   arma::vec d1ac = d1AC["grad"];
   arma::umat iA = d1AC["iA"];
   arma::ucube iC = d1AC["iC"];
   arma::mat Anew(arma::size(Aold));
   arma::cube Cnew(arma::size(Cold));
   const int p = Aold.n_rows;
   const int q = Cold.n_slices;
   for(int i = 0; i < p; i ++){
      arma::uvec idA = iA.row(i).t();
      arma::vec tA = Aold.row(i).t() % arma::exp(ss*d1ac(idA));
      const double l1A = arma::sum(arma::abs(tA));
      Anew.row(i) = arma::clamp(tA.t() / l1A, arma::datum::eps, 1.0-arma::datum::eps);
      for(int j = 0; j < q; j++){
         arma::uvec idC = iC.slice(j).row(i).t();
         arma::vec tC = Cold.slice(j).row(i).t() % arma::exp(ss*d1ac(idC));
         const double l1C = arma::sum(arma::abs(tC));
         Cnew.slice(j).row(i) = arma::clamp(tC.t() / l1C, arma::datum::eps, 1.0-arma::datum::eps);
      }
   }
   return Rcpp::List::create(Rcpp::Named("A") = Anew,
                             Rcpp::Named("C") = Cnew);
}

Rcpp::List newAD_MD(Rcpp::List& d1AD,arma::mat& Aold, arma::cube& Dold, double ss){ // , arma::vec& d2adV
   arma::vec d1ad = d1AD["gs"];
   // arma::mat d2ad = d1AD["hs"];
   // arma::vec d2adV = -arma::diagvec(d2ad);
   arma::umat iA = d1AD["iA"];
   arma::ucube iD = d1AD["iD"];
   arma::mat Anew(arma::size(Aold));
   arma::cube Dnew(arma::size(Dold));
   const int p = Aold.n_rows;
   const int q = Dold.n_slices;
   for(int i = 0; i < p; i ++){
      arma::uvec idA = iA.row(i).t();
      arma::vec tA = Aold.row(i).t() % arma::exp(ss*d1ad(idA));
      const double l1A = arma::sum(arma::abs(tA));
      Anew.row(i) = arma::clamp(tA.t() / l1A, arma::datum::eps, 1.0-arma::datum::eps);
      for(int j = 0; j < q; j++){
         if(Aold(i,j) > std::sqrt(arma::datum::eps)){
            arma::uvec idD = iD.slice(j).row(i).t();
            // arma::vec tD = Dold.slice(j).row(i).t() + ss*d1ad(idD)/(d2adV(idD) + std::pow(arma::datum::eps,0.5));
            arma::vec tD = Dold.slice(j).row(i).t() + ss*d1ad(idD);
            Dnew.slice(j).row(i) = tD.t();
         } else {
            Dnew.slice(j).row(i) = Dold.slice(j).row(i);
         }
      }
   }
   return Rcpp::List::create(Rcpp::Named("A") = Anew,
                             Rcpp::Named("D") = Dnew);
}

Rcpp::List newAD_MD_adam(Rcpp::List& d1AD,arma::mat& Aold, arma::cube& Dold, double ss,
                         int iter, arma::vec& mt, arma::vec& vt, Rcpp::List& control){
   arma::vec d1ad = d1AD["gs"];
   double b1 = control["adam.b1"];
   double b2 = control["adam.b2"];
   arma::vec mtN = b1*mt + (1-b1)*d1ad;
   arma::vec vtN = b2*vt + (1-b2)*arma::pow(d1ad,2);
   arma::vec d1adADAM = (mtN/(1-std::pow(b1,iter))) / (arma::pow(vtN/(1-std::pow(b2,iter)),0.5) + std::sqrt(arma::datum::eps));
   arma::umat iA = d1AD["iA"];
   arma::ucube iD = d1AD["iD"];
   arma::mat Anew(arma::size(Aold));
   arma::cube Dnew(arma::size(Dold));
   const int p = Aold.n_rows;
   const int q = Dold.n_slices;
   for(int i = 0; i < p; i ++){
      arma::uvec idA = iA.row(i).t();
      arma::vec tA = Aold.row(i).t() % arma::exp(ss*d1ad(idA));
      const double l1A = arma::sum(arma::abs(tA));
      Anew.row(i) = arma::clamp(tA.t() / l1A, arma::datum::eps, 1.0-arma::datum::eps);
      for(int j = 0; j < q; j++){
         if(Aold(i,j) > std::sqrt(arma::datum::eps)){
            arma::uvec idD = iD.slice(j).row(i).t();
            arma::vec tD = Dold.slice(j).row(i).t() + ss*d1adADAM(idD);
            Dnew.slice(j).row(i) = tD.t();
         } else {
            Dnew.slice(j).row(i) = Dold.slice(j).row(i);
         }
      }
   }
   mt = mtN;
   vt = vtN;
   return Rcpp::List::create(Rcpp::Named("A") = Anew,
                             Rcpp::Named("D") = Dnew);
}

arma::vec newM(arma::vec& mu, arma::mat& R, arma::mat& Z, double ss){
   const int np = mu.size();
   const int n = Z.n_rows;
   arma::vec gM(np, arma::fill::zeros);
   arma::mat iR = arma::inv_sympd(R, arma::inv_opts::allow_approx);
   for(int i = 0; i < n; i++){
      gM += iR*(Z.row(i) - mu.t()).t();
   }
   mu += ss*gM;
   return(mu);
}

arma::mat newL(arma::vec& mu, arma::mat& L, arma::mat& Z, double ss, bool cor){
   const int n = Z.n_rows;
   const int q = L.n_rows;
   arma::uvec idL = arma::trimatl_ind(arma::size(L));
   arma::vec vL = L(idL);
   arma::uvec iL;
   if(cor){
      iL = arma::regspace<arma::uvec>(1,idL.n_elem-1);
   } else {
      iL = arma::regspace<arma::uvec>(0,idL.n_elem-1);
   }
   idL = idL(iL);
   vL = vL(iL);
   const int np = vL.size();
   arma::vec gL(np);
   arma::mat Ri = L*L.t();
   arma::mat iR = arma::inv_sympd(Ri, arma::inv_opts::allow_approx);

   for(int i = 0; i < np; i++){
      arma::uword iLi = idL(i);
      arma::mat Djk(q,q);
      Djk(iLi) = 1;
      arma::mat Gjk = iR*Djk*L.t()*iR;
      double g1 = -n*arma::trace(L.t()*iR*Djk);
      double g2 = 0;
      for(int m = 0; m < n; m++){
         g2 += arma::as_scalar((Z.row(m)-mu.t())*Gjk*(Z.row(m)-mu.t()).t());
      }
      gL(i) = g1 + g2;
   }
   vL += ss*gL;
   L(idL) = vL;
   if(cor){
      L = ProxL(L);
   }
   return(L);
}

arma::mat newZ_ULA(arma::mat& Y, arma::mat& PI,arma::mat& Z, arma::mat& A, arma::cube& C,
                   arma::vec& mu, arma::mat& R,
                   arma::mat& isd, double& h, arma::vec& knots, const unsigned int degree){
   arma::mat YmPIo = (Y - PI)/(PI % (1-PI));
   const int n = YmPIo.n_rows;
   const int q = R.n_cols;
   arma::mat gpz = d1PostZ(YmPIo,Z,isd,A,C,mu,R);
   arma::mat rE = rmvStNorm(n,q);
   arma::mat Zn = Z + h*gpz + std::sqrt(2*h)*rE;
   return(Zn);
}

arma::mat newZ_MALA(arma::mat& Y, arma::mat& PI,arma::mat& Z, arma::mat& A, arma::cube& C,
                    arma::vec& mu, arma::mat& R, double& ar,
                    arma::mat& isd, double& h, arma::vec& knots, const unsigned int degree,
                    const std::string& basis){
   arma::mat YmPIo = (Y - PI)/(PI % (1-PI));
   const int n = YmPIo.n_rows;
   const int q = R.n_cols;
   const int np = knots.size() + degree ;
   arma::mat spM(n,q*np), spD(n,q*np);
   arma::cube spObj(n,q*np,2);

   arma::mat gpz = d1PostZ(YmPIo,Z,isd,A,C,mu,R);
   arma::mat rE = rmvStNorm(n,q);
   arma::mat Zn = Z + h*gpz + std::sqrt(2*h)*rE;

   arma::mat Un = Z2U(Zn);
   if(basis == "is"){
      spObj = SpU_isp(Un,knots,degree);
      spM = spObj.slice(0);
      spD = spObj.slice(1);
   } else {
      spObj = SpU_bsp(Un,knots,degree);
      spM = spObj.slice(0);
      spD = spObj.slice(1);
   }
   arma::mat PIn = prob(A,C,spM);
   arma::vec pdo = arma::sum(fyz(Y,PI),1) + arma::sum(fz(Z,mu,R),1);
   arma::vec pdn = arma::sum(fyz(Y,PIn),1) + arma::sum(fz(Zn,mu,R),1);
   arma::mat YmPIn = (Y - PIn)/(PIn % (1-PIn));
   arma::mat gpzn = d1PostZ(YmPIn,Zn,spD,A,C,mu,R);
   arma::vec qo = -1/(4*h) * arma::pow(arma::vecnorm(Z - Zn - h*gpzn,2,1),2);
   arma::vec qn = -1/(4*h) * arma::pow(arma::vecnorm(Zn - Z - h*gpz,2,1),2);
   arma::vec rejV = pdn + qo - pdo - qn;

   arma::mat Zout(arma::size(Z));
   for(int i = 0; i < n; i++){
      double rej = R::runif(0,1);
      double com = std::min(1.0, std::exp(rejV(i)));
      if(rej < com){
         Zout.row(i) = Zn.row(i);
         ar++;
      } else {
         Zout.row(i) = Z.row(i);
      }
   }
   return Zout;
}

arma::mat newZ_RWMH(arma::mat& Y, arma::mat& PI,arma::mat& Z, arma::mat& A, arma::cube& C,
                    arma::vec& mu, arma::mat& R, double& ar,
                    const double h, arma::vec& knots, const unsigned int degree, const std::string& basis){
   const int n = Y.n_rows;
   const int q = R.n_cols;
   const int np = knots.size() + degree ;
   arma::mat spM(n,q*np);
   arma::cube spObj(n,q*np,2);

   arma::mat hD = h*arma::eye(arma::size(R));
   arma::mat rE = rmvNorm(n,mu,hD);
   arma::mat Zn = Z + rE;

   arma::mat Un = Z2U(Zn);
   if(basis == "is"){
      spObj = SpU_isp(Un,knots,degree);
      spM = spObj.slice(0);
   } else {
      spObj = SpU_bsp(Un,knots,degree);
      spM = spObj.slice(0);
   }
   arma::mat PIn = prob(A,C,spM);
   arma::vec pdo = arma::sum(fyz(Y,PI),1) + arma::sum(fz(Z,mu,R),1);
   arma::vec pdn = arma::sum(fyz(Y,PIn),1) + arma::sum(fz(Zn,mu,R),1);

   arma::mat Zout(arma::size(Z));
   for(int i = 0; i < n; i++){
      double rej = R::runif(0,1);
      double com = std::min(1.0,std::exp(pdn(i) - pdo(i)));
      if(rej < com){
         Zout.row(i) = Zn.row(i);
         ar++;
      } else {
         Zout.row(i) = Z.row(i);
      }
   }
   return Zout;
}

arma::mat newG_MD(Rcpp::List& d1G, arma::mat& Gold, double ss){
   arma::vec d1g = d1G["grad"];
   arma::umat iG = d1G["iG"];
   arma::mat Gnew(arma::size(Gold));
   const int p = Gold.n_rows;
   for(int i = 0; i < p; i++){
      arma::uvec ig = arma::find(Gold.row(i) != 0);
      arma::uvec idG = iG.row(i).t();
      idG = idG(ig);
      arma::vec Goi = Gold.row(i).t();
      Goi = Goi(ig);
      arma::vec tG = Goi % arma::exp(ss*d1g(idG));
      const double l1G = arma::sum(arma::abs(tG));
      arma::vec Gni = arma::clamp(tG / l1G, arma::datum::eps, 1.0-arma::datum::eps);
      arma::vec tGi(Gnew.n_cols);
      tGi(ig) = Gni;
      Gnew.row(i) = tGi.t();
   }
   return(Gnew);
}

arma::mat newZ_ULA_aCDM(arma::mat& Y, arma::mat& PI, arma::mat& Z, arma::mat& Qmatrix, arma::mat& Apat,
                        arma::mat& G, arma::vec& mu, arma::mat& R, double& h){
   const int n = Y.n_rows;
   const int q = mu.n_elem;
   arma::mat gpz = d1PostZ_aCDM(Y,PI,Z,Qmatrix,Apat,G,mu,R);
   arma::mat rE = rmvStNorm(n,q);
   arma::mat Zn = Z + h*gpz + std::sqrt(2*h)*rE;
   return(Zn);
}

arma::mat newZ_RWMH_aCDM(arma::mat& Y, arma::mat& PI, arma::mat& Z,
                         arma::mat& G, arma::mat& Apat,
                         arma::vec& mu, arma::mat& R, const double h, double& ar){
   const int n = Y.n_rows;

   arma::mat hD = h*arma::eye(arma::size(R));
   arma::mat rE = rmvNorm(n,mu,hD);
   arma::mat Zn = Z + rE;
   arma::mat Un = Z2U(Zn);

   arma::mat PIn = prob_aCDM(G,Un,Apat);
   arma::vec pdo = arma::sum(fyz(Y,PI),1) + arma::sum(fz(Z,mu,R),1);
   arma::vec pdn = arma::sum(fyz(Y,PIn),1) + arma::sum(fz(Zn,mu,R),1);

   arma::mat Zout(arma::size(Z));
   for(int i = 0; i < n; i++){
      double rej = R::runif(0,1);
      double com = std::min(1.0,std::exp(pdn(i) - pdo(i)));
      if(rej < com){
         Zout.row(i) = Zn.row(i);
         ar++;
      } else {
         Zout.row(i) = Z.row(i);
      }
   }
   return Zout;
}

arma::mat newZ_MALA_aCDM(arma::mat& Y, arma::mat& PI, arma::mat& Z, arma::mat& Qmatrix, arma::mat& Apat,
                         arma::mat& G, arma::vec& mu, arma::mat& R, double& h, double& ar){
   const int n = Y.n_rows;
   const int q = mu.n_elem;

   arma::mat gpz = d1PostZ_aCDM(Y,PI,Z,Qmatrix,Apat,G,mu,R);
   arma::mat rE = rmvStNorm(n,q);
   arma::mat Zn = Z + h*gpz + std::sqrt(2*h)*rE;
   arma::mat Un = Z2U(Zn);

   arma::mat PIn = prob_aCDM(G,Un,Apat);
   arma::vec pdo = arma::sum(fyz(Y,PI),1) + arma::sum(fz(Z,mu,R),1);
   arma::vec pdn = arma::sum(fyz(Y,PIn),1) + arma::sum(fz(Zn,mu,R),1);

   arma::mat gpzn = d1PostZ_aCDM(Y,PIn,Zn,Qmatrix,Apat,G,mu,R);
   arma::vec qo = -1/(4*h) * arma::pow(arma::vecnorm(Z - Zn - h*gpzn,2,1),2);
   arma::vec qn = -1/(4*h) * arma::pow(arma::vecnorm(Zn - Z - h*gpz,2,1),2);
   arma::vec rejV = pdn + qo - pdo - qn;

   arma::mat Zout(arma::size(Z));
   for(int i = 0; i < n; i++){
      double rej = R::runif(0,1);
      double com = std::min(1.0, std::exp(rejV(i)));
      if(rej < com){
         Zout.row(i) = Zn.row(i);
         ar++;
      } else {
         Zout.row(i) = Z.row(i);
      }
   }
   return Zout;
}
