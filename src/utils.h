#ifndef UTILS_H
#define UTILS_H

double dCop(arma::vec& z, arma::mat& R, const bool log = true);

arma::mat Z2U(arma::mat& Z);

arma::cube D2C(arma::cube& D);

arma::mat cumsumMat(arma::vec& expd);

arma::vec ProxD(arma::vec& y);

arma::mat ProxL(arma::mat& L);

arma::mat rmvNorm(const int n, arma::vec& mu, arma::mat& R);

arma::mat rmvStNorm(const int n, const int q);

double dmvNorm(arma::vec& y, arma::vec& mu, arma::mat& R, const bool log = true);

arma::mat dmvStNorm(arma::mat& Z);

Rcpp::List genpar(const int p, const int q,
                  const double probSparse,
                  arma::vec& knots, const int degree, const std::string& basis);

Rcpp::List SpU(arma::mat& U, arma::vec& knots, const unsigned int deg, const std::string& basis);

arma::cube SpU_isp(arma::mat& U, arma::vec& knots, const unsigned int deg);

arma::cube SpU_bsp(arma::mat& U, arma::vec& knots, const unsigned int deg);

arma::mat prob(arma::mat& A, arma::cube& C, arma::mat& ism);

arma::mat fyz(arma::mat& Y, arma::mat& PI);

arma::mat fz(arma::mat& Z, arma::mat& R);

double fy(arma::mat& Y, arma::mat& A, arma::cube& C,
          arma::vec& mu, arma::mat& R, Rcpp::List& control);

Rcpp::List d1AC(arma::mat& Y, arma::mat& PI, arma::mat& ism,
               arma::mat& A, arma::cube& C);

arma::mat d1CdD(arma::vec& d);

Rcpp::List d1AD(arma::mat& Y, arma::mat& PI, arma::mat& ism,
                arma::mat& A, arma::cube& D);

Rcpp::List d2AC(arma::mat& YmPI, arma::mat& ism,
                arma::mat& A, arma::cube& C);

arma::cube d1PIdZ(arma::mat& A, arma::cube& C, arma::mat& isd, arma::mat& Z);

arma::mat d1PostZ(arma::mat& YmPI, arma::mat& Z, arma::mat& isd,
                 arma::mat& A, arma::cube& C, arma::vec& mu, arma::mat& R);

#endif
