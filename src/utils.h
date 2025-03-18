#ifndef UTILS_H
#define UTILS_H

// double dCop(arma::vec& z, arma::mat& R, const bool log = true);

arma::mat Z2U(arma::mat& Z);

arma::cube D2C(arma::cube& D);

arma::mat cumsumMat(arma::vec& expd);

arma::vec ProxD(arma::vec& y);

arma::mat ProxL(arma::mat& L);

arma::mat rmvNorm(const int n, arma::vec& mu, arma::mat& R);

arma::mat rmvStNorm(const int n, const int q);

double dmvNorm(arma::vec& y, arma::vec& mu, arma::mat& R, const bool log = true);

arma::mat dmvStNorm(arma::mat& Z);

// double UiAk(arma::rowvec& Ui, arma::rowvec& Ak);

arma::mat UA(arma::mat& U, arma::mat& Apat);

// arma::rowvec dUiAk(arma::rowvec& Ui, arma::rowvec& Ak);

arma::mat dUA(arma::mat& U, arma::rowvec& Ak);

Rcpp::List genpar(const int p, const int q,
                  const double probSparse,
                  arma::vec& knots, const int degree, const std::string& basis);

arma::mat genpar_aCDM(arma::mat& Qmatrix, const double maxG0);

arma::cube SpU_isp(arma::mat& U, arma::vec& knots, const unsigned int deg);

arma::cube SpU_bsp(arma::mat& U, arma::vec& knots, const unsigned int deg);

arma::mat prob(arma::mat& A, arma::cube& C, arma::mat& ism);

arma::mat fyz(arma::mat& Y, arma::mat& PI);

arma::mat fz(arma::mat& Z, arma::vec& mu, arma::mat& R);

double fy_gapmCDM(arma::mat& Y, arma::mat& A, arma::cube& C,
                  arma::vec& mu, arma::mat& R, Rcpp::List& control);

double fy_gapmCDM_IS(arma::mat& Y, arma::mat& A, arma::cube& C,
                     arma::vec& mu, arma::mat& R,
                     arma::rowvec& pmur, arma::mat& pR, Rcpp::List& control);

Rcpp::List d1AC(arma::mat& Y, arma::mat& PI, arma::mat& ism,
                arma::mat& A, arma::cube& C);

arma::mat d1CdD(arma::vec& d);

Rcpp::List d1AD(arma::mat& Y, arma::mat& PI, arma::mat& ism,
                arma::mat& A, arma::cube& D);

// Rcpp::List d2AC(arma::mat& YmPI, arma::mat& ism,
//                 arma::mat& A, arma::cube& C);

arma::cube d1PIdZ(arma::mat& A, arma::cube& C, arma::mat& isd, arma::mat& Z);

arma::cube d1PIdZ_aCDM(arma::mat& G, arma::mat& Qmatrix, arma::mat& Z, arma::mat& Apat);

arma::mat d1PostZ(arma::mat& YmPI, arma::mat& Z, arma::mat& isd,
                 arma::mat& A, arma::cube& C, arma::vec& mu, arma::mat& R);

// Rcpp::List aCDM(arma::mat& G, arma::mat& Qmatrix, arma::mat& Z, arma::mat& Apat);

arma::mat prob_aCDM(arma::mat& G, arma::mat& U, arma::mat& Apat);

double fy_aCDM(arma::mat& Y, arma::mat& G, arma::mat& Qmatrix, arma::mat& Apat,
               arma::vec& mu, arma::mat& R, Rcpp::List& control);

double fy_aCDM_IS(arma::mat& Y, arma::mat& G, arma::mat& Qmatrix, arma::mat& Apat,
                  arma::vec& mu, arma::mat& R,
                  arma::rowvec& pmur, arma::mat& pR, Rcpp::List& control);

// Rcpp::List d1G(arma::mat& Y, Rcpp::List aCDMlist);

Rcpp::List d1G(arma::mat& Y, arma::mat& U, arma::mat& PI, arma::mat& G,
               arma::mat& Apat, arma::mat& Qmatrix);

// arma::mat d1PostZ_aCDM(arma::mat& Y, arma::mat& Z, Rcpp::List aCDMlist, arma::vec& mu, arma::mat& R);

arma::mat d1PostZ_aCDM(arma::mat& Y, arma::mat& PI, arma::mat& Z, arma::mat& Qmatrix, arma::mat& Apat,
                       arma::mat& G, arma::vec& mu, arma::mat& R);

#endif
