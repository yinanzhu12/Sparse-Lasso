# This script process the svm file into pieces of .rds file and divide them into training and
# test data


library(readr)
library(Matrix)
source('~/GitHub/Sparse-Lasso/gradient descent functions.R')

base_dir = "~/url_svmlight/"
svm_files = dir(base_dir, pattern = "*.svm")

# Loop through the files and create a list of objects
for(i in seq_along(svm_files)) {
  myfile = svm_files[i]
  cat(paste0("Reading file ", i, ": ", myfile, "\n"))
  D = read_svmlight_class(paste0(base_dir, myfile), num_cols = 3231962)
  if(i<=length(seq_along(svm_files))/2){
    if(i == 1){
      Xtrain = D$features
      ytrain = D$labels
    }
    else{
      Xtrain = rBind(Xtrain,D$features)
      ytrain = c(ytrain,D$labels)
    }
  }
  else{
    if(i == length(seq_along(svm_files))/2+1){
      Xtest = D$features
      ytest = D$labels
    }
    else{
      Xtest = rBind(Xtest,D$features)
      ytest = c(ytest,D$labels)
    }
  }
}

Xtrain = cBind(Xtrain,1)
Xtest = cBind(Xtest,1)
ytrain=ytrain+{ytrain == -1}
ytest = ytest + {ytest == -1}
# Save as serialized (binary) files for much faster read-in next time
saveRDS(Xtrain, file='url_Xtrain.rds')
saveRDS(ytrain, file='url_ytrain.rds')
saveRDS(Xtest, file='url_Xtest.rds')
saveRDS(ytest, file='url_ytest.rds')