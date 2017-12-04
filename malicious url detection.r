#We use 496000 observations as training data and 480130 as test data

library(readr)
library(Matrix)
library(microbenchmark)
library(Rcpp)
library(RcppEigen)
Rcpp::sourceCpp('gradient_descent_functions.cpp')
source('~/GitHub/SDS385-course-work/Excercise 4 Dealing with large data set/malicious url detection/gradient descent functions.R')
Xtest=readRDS('~/url_svmlight/url_Xtest.rds')
ytest=readRDS('~/url_svmlight/url_ytest.rds')
Xtrain=readRDS('~/url_svmlight/url_Xtrain.rds')
Xtrain=t(Xtrain)
ytrain=readRDS('~/url_svmlight/url_ytrain.rds')


#generate a permutation of samples by which we will update stochastic gradient descent

rn = sample(length(ytrain))


#first, a sgd without regularization, it runs 1 epoch at 1.3 s

epoch=1
microbenchmark(sgdC(rn,Xtrain,ytrain,epoch),times=1L)


epoch=100
result=sgdC(rn,Xtrain,ytrain,epoch)


nllh=result[1:epoch]
plot(nllh,type='l',xlab='epoch',ylab='target function',sub='convergence of sgd without lasso')

#make prediction on the test data set

beta=result[(epoch+1):length(result)]
og=omega(Xtest,beta)
ypredict=1*{og>0.5}
dif=ypredict-ytest


#true positive~0.285

sum((1*{dif==0})*(1*{ypredict==1}))/length(ytest)

#true negative~0.695

sum((1*{dif==0})*(1*{ypredict==0}))/length(ytest)

#false positive~0.007

sum(1*{dif==1})/length(ytest)


#false negative~0.01

sum(1*{dif==-1})/length(ytest)



#sgd with lasso lazy update,runs 1 epoch for 1.61 s

epoch=1
lambda=0.000001
microbenchmark(sgdC_lasso(rn,Xtrain,ytrain,epoch,lambda),times=1L)


epoch=100
result_lasso=sgdC_lasso(rn,Xtrain,ytrain,epoch,lambda)


nllh_lasso=result_lasso[1:epoch]
plot(nllh_lasso,type='l',xlab='epoch',ylab='target function',sub='sgd with lasso and lazy update')

beta=result_lasso[(epoch+1):length(result_lasso)]
og=omega(Xtest,beta)
ypredict=1*{og>0.5}
dif=ypredict-ytest

#true positive~0.288
```{r}
sum((1*{dif==0})*(1*{ypredict==1}))/length(ytest)
```
#true negative~0.694
```{r}
sum((1*{dif==0})*(1*{ypredict==0}))/length(ytest)
```
#false positive~0.008
```{r}
sum(1*{dif==1})/length(ytest)
```

#false negative~0.008
```{r}
sum(1*{dif==-1})/length(ytest)
```
