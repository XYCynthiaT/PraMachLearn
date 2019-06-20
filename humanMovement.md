---
title: "Human Movements Analysis"
author: "Cynthia Tang"
date: "June 18, 2019"
output: 
  html_document:
    keep_md: true
---



## Downloading and Reading in Data  

```r
if(!dir.exists("data")) dir.create("data")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
              "data/training.csv", mode = "wb")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
              "data/testing.csv", mode = "wb")
```


```r
training <- read.csv("data/training.csv", na.strings = c("", NA))
testing <- read.csv("data/testing.csv", na.strings = c("", NA))
library(caret)
```

## Tidy Data  
Removing columns with NAs.  

```r
NAs <- apply(training, 2, function(x) sum(is.na(x)))
iNAs <- which(NAs > 0)
training <- training[, -iNAs]
testing <- testing[, -iNAs]
# str(training)
```

There are still 60 variables including both continuous and categorical.  
  
Removing zero covariates.  

```r
nzv <- nearZeroVar(training, saveMetrics = TRUE)
inzv <- which(nzv[,4] == TRUE)
training <- training[, -inzv]
testing <- testing[, -inzv]
# str(training)
```

Removing the first five variables unrelated to the movements.  

```r
trainingPara <- training[, -c(1:5)]
testingPara <- testing[, -c(1:5)]
ncol(trainingPara); ncol(testingPara)
```

```
## [1] 54
```

```
## [1] 54
```

Correlation.  

```r
correl <- abs(cor(trainingPara[,-54]))
diag(correl) <- 0
corVar <- which(correl > 0.8, arr.ind = TRUE)
nrow(corVar)
```

```
## [1] 38
```

Now, we get the cleaned training and testing data with 53 predictors and 38 of 
these predictors are highly correlated. Thus, we chose to preprocess with 
Principle Component Analysis.   

## Data Slicing  

We split training data into training and testing subsets. 70% training data are
assigned to training subset.

```r
set.seed(3753)
inTrain <- createDataPartition(trainingPara$classe, p = 0.70, list = FALSE)
subtraining <- trainingPara[inTrain,]
subtesting <- trainingPara[-inTrain,]
dim(subtraining);dim(subtesting)
```

```
## [1] 13737    54
```

```
## [1] 5885   54
```

## Data Modeling  
  
We fit a predictive model using __random forest__ algorithm because it fits a non-linear relationship and automatically selects the optimal predictors. We will use __10-fold cross validation__ when applying the algorithm. We expect the error rate should below 5%.


```r
# random forest
# set.seed(2314)
# modFit1 <- randomForest::randomForest(classe ~., data = subtraining, 
#                  trControl = trainControl(method="cv", 10))
# modFit1
# caret
set.seed(2314)
modFit2 <- train(classe ~., data = subtraining, 
                 trControl = trainControl(method="cv", 10))
modFit2
```

```
## Random Forest 
## 
## 13737 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12362, 12362, 12366, 12364, 12365, 12363, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9936668  0.9919889
##   27    0.9970149  0.9962242
##   53    0.9951954  0.9939225
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

```r
modFit2$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.24%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    2    0    0    0 0.0005120328
## B    7 2648    3    0    0 0.0037622272
## C    0    4 2390    2    0 0.0025041736
## D    0    0   10 2241    1 0.0048845471
## E    0    0    0    4 2521 0.0015841584
```

Then, we estimate the performance of the model on testing data.  

```r
# pred1 <- predict(modFit1, newdata = subtesting)
# confusionMatrix(pred1, subtesting$classe)

pred2 <- predict(modFit2, newdata = subtesting)
confusionMatrix(pred2, subtesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    1    0    0    0
##          B    0 1136    3    0    0
##          C    0    2 1020    3    0
##          D    0    0    3  961    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.998           
##                  95% CI : (0.9964, 0.9989)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9974          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9942   0.9969   1.0000
## Specificity            0.9998   0.9994   0.9990   0.9994   1.0000
## Pos Pred Value         0.9994   0.9974   0.9951   0.9969   1.0000
## Neg Pred Value         1.0000   0.9994   0.9988   0.9994   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1930   0.1733   0.1633   0.1839
## Detection Prevalence   0.2846   0.1935   0.1742   0.1638   0.1839
## Balanced Accuracy      0.9999   0.9984   0.9966   0.9981   1.0000
```

The accuracy of the model is 0.9979609.  

## Prediction  
We apply the model to the testing data set.  

```r
predV <- predict(modFit2, testingPara)
predV
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


