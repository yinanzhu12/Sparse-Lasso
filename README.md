# Sparse-Lasso
We present and implement a lazy update strategy to fit lasso logistic classification model on the URL Reputation Data Set http://archive.ics.uci.edu/ml/datasets/URL+Reputation

The lazy update is explained in the pdf file. Aside from the that. We used various ingradient in the implementation:

1. Stochastic gradient descent: We resample all the features and consider one feature at one time.
2. Adagrad: Instead of naive gradient descent, we use quasi-newton method to approach minimum faster. The latter can be further optimized by Adagrad (https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
3. Rcpp: A package (http://adv-r.had.co.nz/Rcpp.html) that allow one to run C++ functions in R. Its sub-package RcppEigen also provides nice treatment to sparse matrices
