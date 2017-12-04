# This script process the svm file into pieces of .rds file and divide them into training and
# test data


library(readr)
library(Matrix)

read_svmlight_class = function(myfile, format='sparseMatrix', num_cols = NULL) {
  require(Matrix)
  require(readr)
  
  raw_x = read_lines(myfile)
  x = strsplit(raw_x, ' ', fixed=TRUE)
  x = lapply(x, function(y) strsplit(y, ':', fixed=TRUE))
  l = lapply(x, function(y) as.numeric(unlist(y)))
  label = as.integer(lapply(l, function(x) x[1]))
  num_rows = length(label)
  features = lapply(l, function(x) tail(x,-1L))
  row_length = as.integer(lapply(features, function(x) length(x)/2))
  features = unlist(features)
  i = rep.int(seq_len(num_rows), row_length)
  j = features[seq.int(1, length(features), by = 2)] + 1
  v = features[seq.int(2, length(features), by = 2)]
  
  if(missing(num_cols)) {
    num_cols = max(j)
  }
  m = sparseMatrix(i=i, j=j, x=v, dims=c(num_rows, num_cols))
  
  list(labels=label, features=m)
}

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