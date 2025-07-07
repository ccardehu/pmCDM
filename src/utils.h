#ifndef UTILS_H
#define UTILS_H

arma::mat Z2U(arma::mat& Z);

arma::cube D2C(arma::cube& D);

arma::mat cumsumMat(arma::vec& expd);

arma::vec ProxD(arma::vec& y);

arma::mat ProxL(arma::mat& L);

void cube2eye(arma::cube& C);

arma::mat rmvNorm(const int n, arma::vec& mu, arma::mat& R);

arma::mat rmvNorm_IS(arma::mat& posM, arma::cube& posR);

arma::mat rmvStNorm(const int n, const int q);

double dmvNorm(arma::vec& y, arma::vec& mu, arma::mat& R, const bool log = true);

arma::mat dmvStNorm(arma::mat& Z);

arma::mat UA(arma::mat& U, arma::mat& Apat);

arma::mat dUA(arma::mat& U, arma::rowvec& Ak);

Rcpp::List genpar(const int p, const int q, const int tp,
                  const double probSparse,
                  const std::string& basis);

arma::mat genpar_aCDM(arma::mat& Qmatrix, const double maxG0);

arma::cube SpU_isp(arma::mat& U, arma::vec& knots, const unsigned int deg);

arma::cube SpU_bsp(arma::mat& U, arma::vec& knots, const unsigned int deg);

arma::mat prob(arma::mat& A, arma::cube& C, arma::mat& ism);

arma::mat fyz(arma::mat& Y, arma::mat& PI);

arma::mat fz(arma::mat& Z, arma::vec& mu, arma::mat& R);

arma::mat fz_IS(arma::mat& Z, arma::mat& pM, arma::cube& pR);

double fy_gapmCDM(arma::mat& Y, arma::mat& A, arma::cube& C,
                  arma::vec& mu, arma::mat& R, Rcpp::List& control);

Rcpp::List d1AC(arma::mat& Y, arma::mat& PI, arma::mat& ism,
                arma::mat& A, arma::cube& C);

arma::mat d1CdD(arma::vec& d);

Rcpp::List dCdD(arma::vec& d);

Rcpp::List d1AD(arma::mat& Y, arma::mat& PI, arma::mat& ism,
                arma::mat& A, arma::cube& D); // , arma::vec& gs, arma::vec& vs

arma::cube d1PIdZ(arma::mat& A, arma::cube& C, arma::mat& isd, arma::mat& Z);

arma::cube d1PIdZ_aCDM(arma::mat& G, arma::mat& Qmatrix, arma::mat& Z, arma::mat& Apat);

arma::mat d1PostZ(arma::mat& YmPI, arma::mat& Z, arma::mat& isd,
                 arma::mat& A, arma::cube& C, arma::vec& mu, arma::mat& R);

arma::mat prob_aCDM(arma::mat& G, arma::mat& U, arma::mat& Apat);

double fy_aCDM(arma::mat& Y, arma::mat& G, arma::mat& Qmatrix, arma::mat& Apat,
               arma::vec& mu, arma::mat& R, Rcpp::List& control);

double fy_aCDM_IS(arma::mat& Y, arma::mat& G, arma::mat& Qmatrix, arma::mat& Apat,
                  arma::vec& mu, arma::mat& R, arma::mat& Z,
                  arma::mat& pM, arma::cube& pR, Rcpp::List& control);

Rcpp::List d1G(arma::mat& Y, arma::mat& U, arma::mat& PI, arma::mat& G,
               arma::mat& Apat, arma::mat& Qmatrix);

arma::mat d1PostZ_aCDM(arma::mat& Y, arma::mat& PI, arma::mat& Z, arma::mat& Qmatrix, arma::mat& Apat,
                       arma::mat& G, arma::vec& mu, arma::mat& R);

#endif
