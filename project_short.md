---
title: "Predicting the Manner of Exercise"
output: html_document
---

##Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, the data were collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The goal of this project is to predict the manner in which the participants did the exercise. In this study, we build a prediction model using different features and apply the model to predict 20 different test cases provided.

##Methodology
First, the data was preprocessed to find the relevant variables and format the data so that a classifier could be run on it.

The following steps are taken: 1. Clean the training set. 2. Split it into training/validation sets. 3. Build a model on the training set. 4. Evaluate the model. 5. Apply the model on provided test cases.

**Data Preprocessing**

The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Read data from file

```r
# read the training set
df <-  read.csv("pml-training.csv", na.strings=c("NA",""))
```

The original dataset has 160 variables including the “classe” class variable that indicates the manner of the exercise activity. To reduce dimensionality, only the most useful predictors (i.e., variables) were selected. This was accomplished by eliminating variables that had NAs, non-numeric variables, variables that had too few unique values.


```r
library(caret)

# define a function data frame preprocessing
preproc_df <- function(df){
  
    # Removal of NAs
    clean_df <- df[, which(as.numeric(colSums(is.na(df)))==0)]
    # Removal of Non-numeric Variables
    clean_df <- clean_df[,-(1:7)]
    # Removal of Near-Zero Values
    end <- ncol(clean_df)
    #clean_df[,-end] <- data.frame(sapply(clean_df[,-end], as.numeric))
    nzv <- nearZeroVar(clean_df[, -end], saveMetrics=TRUE)
    clean_df <- clean_df[,!as.logical(nzv$nzv)]
    return(clean_df)
    
}

clean_df <-preproc_df(df)

# look at number of columns of clean data set
dim(clean_df)
```

```
## [1] 19622    53
```

Look at some summary statistics and frequency plot for the “classe” variable.


```r
summary(clean_df$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
plot(clean_df$classe,main = "`classe` frequency plot", xlab = "Types of Weight Lifting Exercices")
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-1.png) 


**Model Building**

First split the training data into training and validation sets.


```r
set.seed(19)
inTrain <- createDataPartition(df$classe, p = 0.6, list = FALSE)
train <- clean_df[inTrain,]
subtest <- clean_df[-inTrain,]
```


Using the features in the training set, we  build our model using the Random Forest. The 2-fold cross validation is used.


```r
ctrl <- trainControl(allowParallel = TRUE, method = "cv", number = 2);
model <- train(classe ~., data = train, method = "rf", trControl=ctrl, importance = TRUE)

model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.73%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3341    6    0    0    1 0.002090800
## B   16 2255    8    0    0 0.010530935
## C    0   11 2038    5    0 0.007789679
## D    0    0   27 1903    0 0.013989637
## E    0    1    6    5 2153 0.005542725
```
The estimate out of sample error is about 0.73%.

**Cross-Validation and Model Evaluation**

Calculate the “out of sample” accuracy which is the prediction accuracy of our model on the validation set.


```r
subtest_pred <- predict(model, subtest)
subtest_error <- confusionMatrix(subtest_pred, subtest$classe)
subtest_error
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2225    8    0    0    0
##          B    6 1506    2    2    0
##          C    1    3 1360   14    4
##          D    0    1    6 1269    4
##          E    0    0    0    1 1434
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9934         
##                  95% CI : (0.9913, 0.995)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9916         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9969   0.9921   0.9942   0.9868   0.9945
## Specificity            0.9986   0.9984   0.9966   0.9983   0.9998
## Pos Pred Value         0.9964   0.9934   0.9841   0.9914   0.9993
## Neg Pred Value         0.9988   0.9981   0.9988   0.9974   0.9988
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2836   0.1919   0.1733   0.1617   0.1828
## Detection Prevalence   0.2846   0.1932   0.1761   0.1631   0.1829
## Balanced Accuracy      0.9977   0.9953   0.9954   0.9926   0.9971
```

From the above result,  the out of sample accuracy value is 99.34%. Therefore the out of sample error is 1- 99.34% = 0.66%.

```r
ose <- 1 - subtest_error$overall[1];
names(ose) <- "Out of Sample Error"
ose
```

```
## Out of Sample Error 
##         0.006627581
```
The out of sample error is similar as the expected value.

**Prediction Assignment**

In this section, we apply our model to each of the 20 test cases in the testing data set provided.


```r
test <- read.csv("pml-testing.csv")
# clean test data
clean_test <- preproc_df(test)
# look at number of columns of clean test set
dim(clean_test)
```

```
## [1] 20 53
```

```r
# predict
answers <- predict(model, clean_test)
answers <- as.character(answers)
answers
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

Finally, we write the answers to files as specified by the course instructor using the following code segment.

```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(answers)
```

##Conclusions

For this project, we chose Random Forest to build our model to predict the manner of exercies acitivities based on the data collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Random Forest builds a highly accurate classifier, which balances bias and variance trade-offs by settling for a balanced model.

We obtained a really good accuracy based on the model we developed above.

