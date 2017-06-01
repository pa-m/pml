Prediction Assignment Writeup
================

Load training and data
----------------------

``` r
library(kernlab)
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:kernlab':
    ## 
    ##     alpha

``` r
library(ggplot2)
```

Set seed for reproducibility.

``` r
set.seed(123)
```

Load pml-traing.csv, remove unwanted columns, and do pca to reduce features number.

``` r
training = read.csv('pml-training.csv')


#prepare data
#remove columns with NA
training=training[,colSums(is.na(training))==0]
#remove columns with empty string
training=training[,colSums(training=='')==0]
#training=training[,names(training)%in% names(testing)]

# pre-preprocessing: remove unwanted columns
prep=function(t) data.frame(
  #X=t$X,u=as.integer(t$user_name),t=as.double(t$raw_timestamp_part_1-min(training$raw_timestamp_part_1))+1e-6*as.double(t$raw_timestamp_part_2),neww=ifelse(t$new_window=='yes',1,0),
  t[,8:dim(training)[2]-1])
train1=prep(training)
# 2nd phase of preprocessing: eliminate features with near-zero variance and do pca to minimize number of features 
pre=preProcess(train1,method=c('nzv','pca'))

train2=data.frame(predict(pre,train1),classe=training$classe)
```

Split our preprocessed data into a training part (train2train) and a test part (test2test), so we can measure training and testing accuracy.

``` r
# split preprocessed dataset to have a test set with outcome, to perform cross validation
inTrain = createDataPartition(training$classe,p=0.9,list=FALSE)
train2train = train2[inTrain,]
train2test =train2[-inTrain,]
featCols=1:dim(train2train)[2]-1

showAccuracy=function (model) {
  train2trainPredict=predict(model,train2train[,featCols])
  train2testPredict=predict(model,train2test[,featCols])
  print('Training accuracy:');print(sum(train2trainPredict==train2train$classe)/dim(train2train)[1]) 
  print('Testing accuracy:');print(sum(train2testPredict==train2test$classe)/dim(train2test)[1])
}
```

Try various classification methods

``` r
# random forest is too long, glm is for binary classifiers only, rpart gives poor accuracy, 
# try rpart
print('Trying rpart...')
```

    ## [1] "Trying rpart..."

``` r
model_rpart=train(train2train[,featCols],train2train$classe,method='rpart')
```

    ## Loading required package: rpart

``` r
showAccuracy(model_rpart) # training:0.39 testing:0.41
```

    ## [1] "Training accuracy:"
    ## [1] 0.3919148
    ## [1] "Testing accuracy:"
    ## [1] 0.3693878

``` r
# try ksvm
print('Trying ksvm...')
```

    ## [1] "Trying ksvm..."

``` r
model_svm=ksvm(classe~.,data=train2train)
showAccuracy(model_svm) # training:0.93 testing:0.91
```

    ## [1] "Training accuracy:"
    ## [1] 0.9330767
    ## [1] "Testing accuracy:"
    ## [1] 0.9367347

``` r
# try lssvm
print('Trying lssvm...')
```

    ## [1] "Trying lssvm..."

``` r
model_lssvm=lssvm(classe~.,data=train2train)
```

    ## Using automatic sigma estimation (sigest) for RBF or laplace kernel

``` r
showAccuracy(model_lssvm) # training:0.69 testing:0.68
```

    ## [1] "Training accuracy:"
    ## [1] 0.695278
    ## [1] "Testing accuracy:"
    ## [1] 0.6938776

``` r
# try boosting
#library(gbm)
#print('Trying boosting...')
#model_gbm = train(classe~.,method='gbm',data=train2train) #too long, aborted. 
#showAccuracy(model_gbm) 
```

``` r
#try bagging
print('Trying bag...')
```

    ## [1] "Trying bag..."

``` r
model_bag=bag(train2train[,featCols],train2train$classe,bagControl=bagControl(fit=ctreeBag$fit,predict = ctreeBag$pred,aggregate = ctreeBag$aggregate,allowParallel = TRUE))
```

    ## Warning: executing %dopar% sequentially: no parallel backend registered

``` r
showAccuracy(model_bag) #training 0.97 testing: 0.91
```

    ## [1] "Training accuracy:"
    ## [1] 0.9690862
    ## [1] "Testing accuracy:"
    ## [1] 0.9341837

Bagging gives the best result, use this method to predict classe for pml-testing.csv.

``` r
# predictions on pml-testing.csv
testing = read.csv('pml-testing.csv')
testing=testing[,names(testing)%in% names(training)]
testingPredict=predict(model_bag,predict(pre,prep(testing))[,featCols])
print('Predictions on pml-testing.csv using best method(bag):')
```

    ## [1] "Predictions on pml-testing.csv using best method(bag):"

``` r
print(data.frame(X=testing$X,testingPredict))
```

    ##     X testingPredict
    ## 1   1              B
    ## 2   2              A
    ## 3   3              A
    ## 4   4              A
    ## 5   5              A
    ## 6   6              C
    ## 7   7              D
    ## 8   8              B
    ## 9   9              A
    ## 10 10              A
    ## 11 11              B
    ## 12 12              B
    ## 13 13              B
    ## 14 14              A
    ## 15 15              E
    ## 16 16              E
    ## 17 17              A
    ## 18 18              B
    ## 19 19              B
    ## 20 20              B

Thanks for reading.
