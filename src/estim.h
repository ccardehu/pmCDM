#ifndef ESTIM_H
#define ESTIM_H

Rcpp::List newAC_PGD(Rcpp::List& d1AC, arma::mat& Aold, arma::cube& Cold, double ss);

Rcpp::List newAD_PGD(Rcpp::List& d1AD, arma::mat& Aold, arma::cube& Dold, double ss);

Rcpp::List newAC_MD(Rcpp::List& d1AC, arma::mat& Aold, arma::cube& Cold, double ss);

Rcpp::List newAD_MD(Rcpp::List& d1AD, arma::mat& Aold, arma::cube& Dold, double ss);

arma::vec newM(arma::vec& mu, arma::mat& R, arma::mat& Z, double ss);

arma::mat newL(arma::vec& mu, arma::mat& L, arma::mat& Z, double ss);

arma::mat newZ_ULA(arma::mat& Y, arma::mat& PI, arma::mat& Z, arma::mat& A, arma::cube& C,
                   arma::vec& mu, arma::mat& R,
                   arma::mat& isd, double& h, arma::vec& knots, const unsigned int degree);

arma::mat newZ_MALA(arma::mat& Y, arma::mat& PI,arma::mat& Z, arma::mat& A, arma::cube& C,
                    arma::vec& mu, arma::mat& R, double& ar,
                    arma::mat& isd, double& h, arma::vec& knots, const unsigned int degree, const std::string& basis);

arma::mat newZ_RWMH(arma::mat& Y, arma::mat& PI,arma::mat& Z, arma::mat& A, arma::cube& C,
                    arma::vec& mu, arma::mat& R, double& ar,
                    const double h, arma::vec& knots, const unsigned int degree, const std::string& basis);

#endif
