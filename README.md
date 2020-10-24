Proactical Machine Learning Prediction Assignment
Libraries
library(caret)
## Warning: package 'caret' was built under R version 3.1.3
## Loading required package: lattice
## Loading required package: ggplot2
library(doParallel)
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
set.seed(20150125)
Loading Training Data
The pml-training.csv data is used to devise training and testing sets during fitting of the model. The pml-test.csv data is used to submit 20 test cases based on the fitted model.

download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', 'pml-training.csv')
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv','pml-test.csv' )
Tidying data
Convert all blank(‘“”’), ‘#DIV/0’ and ‘NA’ values are converted to ‘NA’.

trainingSrc   <- read.csv('pml-training.csv', na.strings=c("NA","#DIV/0!", ""))
testSrc       <- read.csv('pml-test.csv' , na.strings=c("NA", "#DIV/0!", ""))
We decided to leave columns having no more than 60% of NA values:

goodVars    <- which((colSums(!is.na(trainingSrc)) >= 0.6*nrow(trainingSrc)))
trainingSrc <- trainingSrc[,goodVars]
testSrc     <- testSrc[,goodVars]
Some minor fixes to test set are needed to perform well with random forests.

# remove problem id
testSrc <- testSrc[-ncol(testSrc)]
# fix factor levels
testSrc$new_window <- factor(testSrc$new_window, levels=c("no","yes"))
Remove X and cvtd_timestamp colums from the dataset since they are not relevant

trainingSrc <- trainingSrc[,-c(1,5)]
testSrc     <- testSrc[,-c(1,5)]
Partition data into training and test sets
We are dividing data to 60% training and 40% testing sets.

inTraining  <- createDataPartition(trainingSrc$classe, p = 0.6, list = FALSE)
training    <- trainingSrc[inTraining, ]
testing     <- trainingSrc[-inTraining, ]
Fitting Random Forests
The outcome variable is class and other colums are in data dataframe.

class <- training$classe
data  <- training[-ncol(training)]
We will use Parallel Random Forest algorithm to fit the model. Note that for random forests there is no need for cross-validation to get an unbiased estimate of the test set error. It is estimated internally during the fitting process.

registerDoParallel()
rf <- train(data, class, method="parRF", 
    tuneGrid=data.frame(mtry=3), 
    trControl=trainControl(method="none"))
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
rf
## Parallel Random Forest 
## 
## 11776 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: None
Let’s plot importance of the model variables:

plot(varImp(rf))


Confusion Matrix for testing set
Predict on testing set and generate the confusion matrix for the testing set

testingPredictions <- predict(rf, newdata=testing)
confMatrix <- confusionMatrix(testingPredictions,testing$classe)
confMatrix
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230    1    0    0    0
##          B    2 1516    9    0    0
##          C    0    1 1357   16    0
##          D    0    0    2 1270    4
##          E    0    0    0    0 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9955          
##                  95% CI : (0.9938, 0.9969)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9944          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9987   0.9920   0.9876   0.9972
## Specificity            0.9998   0.9983   0.9974   0.9991   1.0000
## Pos Pred Value         0.9996   0.9928   0.9876   0.9953   1.0000
## Neg Pred Value         0.9996   0.9997   0.9983   0.9976   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1932   0.1730   0.1619   0.1833
## Detection Prevalence   0.2843   0.1946   0.1751   0.1626   0.1833
## Balanced Accuracy      0.9995   0.9985   0.9947   0.9933   0.9986
Let’s have a look at the accuracy

confMatrix$overall[1]
##  Accuracy 
## 0.9955391
It looks very good — it is more then 99,5%.

Submit results of Test Set
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

answers <- predict(rf, testSrc)
pml_write_files(answers)
