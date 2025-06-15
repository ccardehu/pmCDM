#' @keywords internal
#' @aliases pmCDM-package
"_PACKAGE"

## usethis namespace: start
#' @importFrom stats rnorm
#' @importFrom methods slot
#' @importFrom mvtnorm rmvnorm
#' @importFrom simstudy genCorMat
#' @importFrom ROCR prediction performance
#' @importFrom psych fa
## usethis namespace: end
NULL

## usethis namespace: start
#' @useDynLib pmCDM, .registration = TRUE
## usethis namespace: end
NULL
