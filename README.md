# ProjectAssignment6

## Practical Machine Learning: Course project


This repository contains the R code and the documentation of the final course project of the MOOC [Practical Machine Learning](https://www.coursera.org/course/predmachlearn) on Coursera.

The task was to predict the type of barbell lift based on data from several acceleromete

The corresponding R commands can be found in the RMarkdown file [`writeup.Rmd`](https://github.com/shravishthaak/ProjectAssignment6/blob/master/writeup.Rmd)
SYNOPSIS

Given both training and test data from the following study:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

the goal of this project, as specified in Professor Leek's instructions, is to “predict the manner in which they did the exercise.”

Further, Professor Leek states that this report should describe:

“how you built your model”
“how you used cross validation”
“what you think the expected out of sample error is”
“why you made the choices you did”
Ultimately, the prediction model is to be run on the test data to predict the outcome of 20 different test cases.

In his second lecture, Professor Leek introduces the “Components of a Predictor” and defines five stages:

Question
Input Data
Features
Algorithm
Parameters
Evaluation
I've decided to proceed along this path.

First, though, I'll load the appropriate packages and set the seed for reproduceable results.

library(AppliedPredictiveModeling)
library(caret)
library(rattle)
library(rpart.plot)
library(randomForest)
QUESTION

In the aforementioned study, six participants participated in a dumbell lifting exercise five different ways. The five ways, as described in the study, were “exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.”

By processing data gathered from accelerometers on the belt, forearm, arm, and dumbell of the participants in a machine learning algorithm, the question is can the appropriate activity quality (class A-E) be predicted?

INPUT DATA

The first step is to import the data and to verify that the training data and the test data are identical.

# Download data.
url_raw_training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
file_dest_training <- "pml-training.csv"
#download.file(url=url_raw_training, destfile=file_dest_training, method="curl")
url_raw_testing <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
file_dest_testing <- "pml-testing.csv"
#download.file(url=url_raw_testing, destfile=file_dest_testing, method="curl")

# Import the data treating empty values as NA.
df_training <- read.csv(file_dest_training, na.strings=c("NA",""), header=TRUE)
colnames_train <- colnames(df_training)
df_testing <- read.csv(file_dest_testing, na.strings=c("NA",""), header=TRUE)
colnames_test <- colnames(df_testing)

# Verify that the column names (excluding classe and problem_id) are identical in the training and test set.
all.equal(colnames_train[1:length(colnames_train)-1], colnames_test[1:length(colnames_train)-1])
## [1] TRUE
FEATURES

Having verified that the schema of both the training and testing sets are identical (excluding the final column representing the A-E class), I decided to eliminate both NA columns and other extraneous columns.

# Count the number of non-NAs in each col.
nonNAs <- function(x) {
    as.vector(apply(x, 2, function(x) length(which(!is.na(x)))))
}

# Build vector of missing data or NA columns to drop.
colcnts <- nonNAs(df_training)
drops <- c()
for (cnt in 1:length(colcnts)) {
    if (colcnts[cnt] < nrow(df_training)) {
        drops <- c(drops, colnames_train[cnt])
    }
}

# Drop NA data and the first 7 columns as they're unnecessary for predicting.
df_training <- df_training[,!(names(df_training) %in% drops)]
df_training <- df_training[,8:length(colnames(df_training))]

df_testing <- df_testing[,!(names(df_testing) %in% drops)]
df_testing <- df_testing[,8:length(colnames(df_testing))]

# Show remaining columns.
colnames(df_training)
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
colnames(df_testing)
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "problem_id"
Professor Leek discusses Level 1 (raw data to covariates) and Level 2 (covariates to new covariates) covariate creation strategies. Given that we're already supplied with the raw sensor data, there's no need for Level 1 processing. However, while being careful not to overfit, some Level 2 processing is certainly worth attempting.

First, check for covariates that have virtually no variablility.

nsv <- nearZeroVar(df_training, saveMetrics=TRUE)
nsv
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt                1.102       6.77811   FALSE FALSE
## pitch_belt               1.036       9.37723   FALSE FALSE
## yaw_belt                 1.058       9.97350   FALSE FALSE
## total_accel_belt         1.063       0.14779   FALSE FALSE
## gyros_belt_x             1.059       0.71348   FALSE FALSE
## gyros_belt_y             1.144       0.35165   FALSE FALSE
## gyros_belt_z             1.066       0.86128   FALSE FALSE
## accel_belt_x             1.055       0.83580   FALSE FALSE
## accel_belt_y             1.114       0.72877   FALSE FALSE
## accel_belt_z             1.079       1.52380   FALSE FALSE
## magnet_belt_x            1.090       1.66650   FALSE FALSE
## magnet_belt_y            1.100       1.51870   FALSE FALSE
## magnet_belt_z            1.006       2.32902   FALSE FALSE
## roll_arm                52.338      13.52563   FALSE FALSE
## pitch_arm               87.256      15.73234   FALSE FALSE
## yaw_arm                 33.029      14.65702   FALSE FALSE
## total_accel_arm          1.025       0.33636   FALSE FALSE
## gyros_arm_x              1.016       3.27693   FALSE FALSE
## gyros_arm_y              1.454       1.91622   FALSE FALSE
## gyros_arm_z              1.111       1.26389   FALSE FALSE
## accel_arm_x              1.017       3.95984   FALSE FALSE
## accel_arm_y              1.140       2.73672   FALSE FALSE
## accel_arm_z              1.128       4.03629   FALSE FALSE
## magnet_arm_x             1.000       6.82397   FALSE FALSE
## magnet_arm_y             1.057       4.44399   FALSE FALSE
## magnet_arm_z             1.036       6.44685   FALSE FALSE
## roll_dumbbell            1.022      83.78351   FALSE FALSE
## pitch_dumbbell           2.277      81.22516   FALSE FALSE
## yaw_dumbbell             1.132      83.14137   FALSE FALSE
## total_accel_dumbbell     1.073       0.21914   FALSE FALSE
## gyros_dumbbell_x         1.003       1.22821   FALSE FALSE
## gyros_dumbbell_y         1.265       1.41678   FALSE FALSE
## gyros_dumbbell_z         1.060       1.04984   FALSE FALSE
## accel_dumbbell_x         1.018       2.16594   FALSE FALSE
## accel_dumbbell_y         1.053       2.37489   FALSE FALSE
## accel_dumbbell_z         1.133       2.08949   FALSE FALSE
## magnet_dumbbell_x        1.098       5.74865   FALSE FALSE
## magnet_dumbbell_y        1.198       4.30129   FALSE FALSE
## magnet_dumbbell_z        1.021       3.44511   FALSE FALSE
## roll_forearm            11.589      11.08959   FALSE FALSE
## pitch_forearm           65.983      14.85577   FALSE FALSE
## yaw_forearm             15.323      10.14677   FALSE FALSE
## total_accel_forearm      1.129       0.35674   FALSE FALSE
## gyros_forearm_x          1.059       1.51870   FALSE FALSE
## gyros_forearm_y          1.037       3.77637   FALSE FALSE
## gyros_forearm_z          1.123       1.56457   FALSE FALSE
## accel_forearm_x          1.126       4.04648   FALSE FALSE
## accel_forearm_y          1.059       5.11161   FALSE FALSE
## accel_forearm_z          1.006       2.95587   FALSE FALSE
## magnet_forearm_x         1.012       7.76679   FALSE FALSE
## magnet_forearm_y         1.247       9.54031   FALSE FALSE
## magnet_forearm_z         1.000       8.57711   FALSE FALSE
## classe                   1.470       0.02548   FALSE FALSE
Given that all of the near zero variance variables (nsv) are FALSE, there's no need to eliminate any covariates due to lack of variablility.

ALGORITHM

We were provided with a large training set (19,622 entries) and a small testing set (20 entries). Instead of performing the algorithm on the entire training set, as it would be time consuming and wouldn't allow for an attempt on a testing set, I chose to divide the given training set into four roughly equal sets, each of which was then split into a training set (comprising 60% of the entries) and a testing set (comprising 40% of the entries).

# Divide the given training set into 4 roughly equal sets.
set.seed(666)
ids_small <- createDataPartition(y=df_training$classe, p=0.25, list=FALSE)
df_small1 <- df_training[ids_small,]
df_remainder <- df_training[-ids_small,]
set.seed(666)
ids_small <- createDataPartition(y=df_remainder$classe, p=0.33, list=FALSE)
df_small2 <- df_remainder[ids_small,]
df_remainder <- df_remainder[-ids_small,]
set.seed(666)
ids_small <- createDataPartition(y=df_remainder$classe, p=0.5, list=FALSE)
df_small3 <- df_remainder[ids_small,]
df_small4 <- df_remainder[-ids_small,]
# Divide each of these 4 sets into training (60%) and test (40%) sets.
set.seed(666)
inTrain <- createDataPartition(y=df_small1$classe, p=0.6, list=FALSE)
df_small_training1 <- df_small1[inTrain,]
df_small_testing1 <- df_small1[-inTrain,]
set.seed(666)
inTrain <- createDataPartition(y=df_small2$classe, p=0.6, list=FALSE)
df_small_training2 <- df_small2[inTrain,]
df_small_testing2 <- df_small2[-inTrain,]
set.seed(666)
inTrain <- createDataPartition(y=df_small3$classe, p=0.6, list=FALSE)
df_small_training3 <- df_small3[inTrain,]
df_small_testing3 <- df_small3[-inTrain,]
set.seed(666)
inTrain <- createDataPartition(y=df_small4$classe, p=0.6, list=FALSE)
df_small_training4 <- df_small4[inTrain,]
df_small_testing4 <- df_small4[-inTrain,]
Based on both the process outlined in Section 5.2 of the aforementioned paper and the concensus in the coursera discussion forums, I chose two different algorithms via the caret package: classification trees (method = rpart) and random forests (method = rf).

PARAMETERS

I decided to try classification trees “out of the box” and then introduce preprocessing and cross validation.

While I also considered applying “out of the box” random forest models, some of the horror stories contributed to the coursera discussion forums regarding the lengthy processing times for random forest models convinced me to only attempt random forests with cross validation and, possibly, preprocessing.

EVALUATION

Classification Tree

First, the “out of the box” classification tree:

# Train on training set 1 of 4 with no extra features.
set.seed(666)
modFit <- train(df_small_training1$classe ~ ., data = df_small_training1, method="rpart")
print(modFit, digits=3)
## CART 
## 
## 2946 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 2946, 2946, 2946, 2946, 2946, 2946, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp      Accuracy  Kappa   Accuracy SD  Kappa SD
##   0.0346  0.531     0.4     0.0355       0.0479  
##   0.0442  0.471     0.308   0.0555       0.0967  
##   0.116   0.324     0.0602  0.0456       0.0641  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0346.
print(modFit$finalModel, digits=3)
## n= 2946 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##   1) root 2946 2110 A (0.28 0.19 0.17 0.16 0.18)  
##     2) roll_belt< 130 2699 1860 A (0.31 0.21 0.19 0.18 0.11)  
##       4) pitch_forearm< -34 220    0 A (1 0 0 0 0) *
##       5) pitch_forearm>=-34 2479 1860 A (0.25 0.23 0.21 0.19 0.12)  
##        10) yaw_belt>=168 138   15 A (0.89 0.072 0 0.036 0) *
##        11) yaw_belt< 168 2341 1780 B (0.21 0.24 0.22 0.2 0.13)  
##          22) magnet_dumbbell_z< -83.5 305  134 A (0.56 0.3 0.046 0.069 0.02) *
##          23) magnet_dumbbell_z>=-83.5 2036 1540 C (0.16 0.23 0.25 0.22 0.14)  
##            46) roll_dumbbell< 57.7 1209  776 C (0.18 0.19 0.36 0.16 0.11) *
##            47) roll_dumbbell>=57.7 827  565 D (0.12 0.29 0.081 0.32 0.19)  
##              94) magnet_belt_y>=590 687  433 D (0.11 0.35 0.07 0.37 0.1)  
##               188) total_accel_dumbbell>=5.5 474  260 B (0.097 0.45 0.1 0.22 0.13) *
##               189) total_accel_dumbbell< 5.5 213   62 D (0.14 0.11 0 0.71 0.042) *
##              95) magnet_belt_y< 590 140   55 E (0.19 0.014 0.14 0.057 0.61) *
##     3) roll_belt>=130 247    1 E (0.004 0 0 0 1) *
fancyRpartPlot(modFit$finalModel)
plot of chunk classification_trees_outofthebox

# Run against testing set 1 of 4 with no extra features.
predictions <- predict(modFit, newdata=df_small_testing1)
print(confusionMatrix(predictions, df_small_testing1$classe), digits=4)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 368  74  11  28   8
##          B  24 151  25  83  30
##          C 135 148 288 138  99
##          D  15   7   0  69   4
##          E  16   0  18   3 219
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5584          
##                  95% CI : (0.5361, 0.5805)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4441          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6595   0.3974   0.8421  0.21495   0.6083
## Specificity            0.9138   0.8975   0.6788  0.98415   0.9769
## Pos Pred Value         0.7526   0.4824   0.3564  0.72632   0.8555
## Neg Pred Value         0.8709   0.8610   0.9532  0.86495   0.9173
## Prevalence             0.2845   0.1938   0.1744  0.16369   0.1836
## Detection Rate         0.1877   0.0770   0.1469  0.03519   0.1117
## Detection Prevalence   0.2494   0.1596   0.4120  0.04844   0.1305
## Balanced Accuracy      0.7866   0.6475   0.7605  0.59955   0.7926
I was very disappointed with the low accuracy rate (0.5584) and hoped for significant improvement by incorporating preprocessing and/or cross validation.

# Train on training set 1 of 4 with only preprocessing.
set.seed(666)
modFit <- train(df_small_training1$classe ~ .,  preProcess=c("center", "scale"), data = df_small_training1, method="rpart")
print(modFit, digits=3)
## CART 
## 
## 2946 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 2946, 2946, 2946, 2946, 2946, 2946, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp      Accuracy  Kappa   Accuracy SD  Kappa SD
##   0.0346  0.531     0.4     0.0355       0.0479  
##   0.0442  0.471     0.308   0.0555       0.0968  
##   0.116   0.324     0.0602  0.0456       0.0641  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0346.
# Train on training set 1 of 4 with only cross validation.
set.seed(666)
modFit <- train(df_small_training1$classe ~ .,  trControl=trainControl(method = "cv", number = 4), data = df_small_training1, method="rpart")
print(modFit, digits=3)
## CART 
## 
## 2946 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2212, 2209, 2208, 2209 
## 
## Resampling results across tuning parameters:
## 
##   cp      Accuracy  Kappa   Accuracy SD  Kappa SD
##   0.0346  0.552     0.427   0.0383       0.0542  
##   0.0442  0.47      0.304   0.0689       0.12    
##   0.116   0.344     0.0914  0.0405       0.061   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0346.
# Train on training set 1 of 4 with both preprocessing and cross validation.
set.seed(666)
modFit <- train(df_small_training1$classe ~ .,  preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data = df_small_training1, method="rpart")
print(modFit, digits=3)
## CART 
## 
## 2946 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2212, 2209, 2208, 2209 
## 
## Resampling results across tuning parameters:
## 
##   cp      Accuracy  Kappa   Accuracy SD  Kappa SD
##   0.0346  0.552     0.427   0.0383       0.0542  
##   0.0442  0.47      0.304   0.0689       0.12    
##   0.116   0.344     0.0914  0.0405       0.061   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0346.
# Run against testing set 1 of 4 with both preprocessing and cross validation.
predictions <- predict(modFit, newdata=df_small_testing1)
print(confusionMatrix(predictions, df_small_testing1$classe), digits=4)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 368  74  11  28   8
##          B  24 151  25  83  30
##          C 135 148 288 138  99
##          D  15   7   0  69   4
##          E  16   0  18   3 219
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5584          
##                  95% CI : (0.5361, 0.5805)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4441          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6595   0.3974   0.8421  0.21495   0.6083
## Specificity            0.9138   0.8975   0.6788  0.98415   0.9769
## Pos Pred Value         0.7526   0.4824   0.3564  0.72632   0.8555
## Neg Pred Value         0.8709   0.8610   0.9532  0.86495   0.9173
## Prevalence             0.2845   0.1938   0.1744  0.16369   0.1836
## Detection Rate         0.1877   0.0770   0.1469  0.03519   0.1117
## Detection Prevalence   0.2494   0.1596   0.4120  0.04844   0.1305
## Balanced Accuracy      0.7866   0.6475   0.7605  0.59955   0.7926
The impact of incorporating both preprocessing and cross validation appeared to show some minimal improvement (accuracy rate rose from 0.531 to 0.552 against training sets). However, when run against the corresponding testing set, the accuracy rate was identical (0.5584) for both the “out of the box” and the preprocessing/cross validation methods.

Random Forest

First I decided to assess the impact/value of including preprocessing.

# Train on training set 1 of 4 with only cross validation.
set.seed(666)
modFit <- train(df_small_training1$classe ~ ., method="rf", trControl=trainControl(method = "cv", number = 4), data=df_small_training1)
print(modFit, digits=3)
## Random Forest 
## 
## 2946 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2212, 2209, 2208, 2209 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.951     0.939  0.00449      0.0057  
##   27    0.955     0.943  0.00582      0.00736 
##   52    0.951     0.938  0.00888      0.0112  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
# Run against testing set 1 of 4.
predictions <- predict(modFit, newdata=df_small_testing1)
print(confusionMatrix(predictions, df_small_testing1$classe), digits=4)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 555  12   1   0   1
##          B   2 358  12   1   0
##          C   0   9 324   6   4
##          D   0   1   5 309   1
##          E   1   0   0   5 354
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9689          
##                  95% CI : (0.9602, 0.9761)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9606          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9946   0.9421   0.9474   0.9626   0.9833
## Specificity            0.9900   0.9905   0.9883   0.9957   0.9963
## Pos Pred Value         0.9754   0.9598   0.9446   0.9778   0.9833
## Neg Pred Value         0.9978   0.9861   0.9889   0.9927   0.9963
## Prevalence             0.2845   0.1938   0.1744   0.1637   0.1836
## Detection Rate         0.2830   0.1826   0.1652   0.1576   0.1805
## Detection Prevalence   0.2902   0.1902   0.1749   0.1611   0.1836
## Balanced Accuracy      0.9923   0.9663   0.9678   0.9792   0.9898
# Run against 20 testing set provided by Professor Leek.
print(predict(modFit, newdata=df_testing))
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
# Train on training set 1 of 4 with only both preprocessing and cross validation.
set.seed(666)
modFit <- train(df_small_training1$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training1)
print(modFit, digits=3)
## Random Forest 
## 
## 2946 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2212, 2209, 2208, 2209 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.951     0.939  0.00382      0.00482 
##   27    0.954     0.942  0.00466      0.0059  
##   52    0.952     0.939  0.0107       0.0135  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
# Run against testing set 1 of 4.
predictions <- predict(modFit, newdata=df_small_testing1)
print(confusionMatrix(predictions, df_small_testing1$classe), digits=4)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 555  10   0   0   0
##          B   2 357  11   0   0
##          C   0  12 327   6   5
##          D   0   1   4 312   1
##          E   1   0   0   3 354
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9714          
##                  95% CI : (0.9631, 0.9784)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9639          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9946   0.9395   0.9561   0.9720   0.9833
## Specificity            0.9929   0.9918   0.9858   0.9963   0.9975
## Pos Pred Value         0.9823   0.9649   0.9343   0.9811   0.9888
## Neg Pred Value         0.9979   0.9855   0.9907   0.9945   0.9963
## Prevalence             0.2845   0.1938   0.1744   0.1637   0.1836
## Detection Rate         0.2830   0.1820   0.1668   0.1591   0.1805
## Detection Prevalence   0.2881   0.1887   0.1785   0.1622   0.1826
## Balanced Accuracy      0.9937   0.9656   0.9710   0.9842   0.9904
# Run against 20 testing set provided by Professor Leek.
print(predict(modFit, newdata=df_testing))
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
Preprocessing actually lowered the accuracy rate from 0.955 to 0.954 against the training set. However, when run against the corresponding set, the accuracy rate rose from 0.9689 to 0.9714 with the addition of preprocessing. Thus I decided to apply both preprocessing and cross validation to the remaining 3 data sets.

# Train on training set 2 of 4 with only cross validation.
set.seed(666)
modFit <- train(df_small_training2$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training2)
print(modFit, digits=3)
## Random Forest 
## 
## 2917 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2188, 2188, 2187, 2188 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.952     0.939  0.00665      0.00844 
##   27    0.954     0.941  0.0102       0.013   
##   52    0.944     0.929  0.00579      0.00735 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
# Run against testing set 2 of 4.
predictions <- predict(modFit, newdata=df_small_testing2)
print(confusionMatrix(predictions, df_small_testing2$classe), digits=4)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 548  11   0   2   0
##          B   3 355  14   1   5
##          C   0   9 323  10   6
##          D   0   1   1 303   5
##          E   1   0   0   2 341
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9634          
##                  95% CI : (0.9541, 0.9713)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9537          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9928   0.9441   0.9556   0.9528   0.9552
## Specificity            0.9906   0.9853   0.9844   0.9957   0.9981
## Pos Pred Value         0.9768   0.9392   0.9282   0.9774   0.9913
## Neg Pred Value         0.9971   0.9866   0.9906   0.9908   0.9900
## Prevalence             0.2844   0.1937   0.1741   0.1638   0.1839
## Detection Rate         0.2823   0.1829   0.1664   0.1561   0.1757
## Detection Prevalence   0.2890   0.1947   0.1793   0.1597   0.1772
## Balanced Accuracy      0.9917   0.9647   0.9700   0.9743   0.9766
# Run against 20 testing set provided by Professor Leek.
print(predict(modFit, newdata=df_testing))
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
# Train on training set 3 of 4 with only cross validation.
set.seed(666)
modFit <- train(df_small_training3$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training3)
print(modFit, digits=3)
## Random Forest 
## 
## 2960 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2219, 2221, 2220, 2220 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.949     0.935  0.00696      0.0088  
##   27    0.951     0.938  0.0105       0.0132  
##   52    0.944     0.929  0.0116       0.0146  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
# Run against testing set 3 of 4.
predictions <- predict(modFit, newdata=df_small_testing3)
print(confusionMatrix(predictions, df_small_testing3$classe), digits=4)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 556  10   0   1   0
##          B   1 357  17   0   4
##          C   1  12 322   7   3
##          D   1   2   2 313   1
##          E   1   0   3   2 354
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9655          
##                  95% CI : (0.9564, 0.9731)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2e-16         
##                                           
##                   Kappa : 0.9563          
##  Mcnemar's Test P-Value : 0.03619         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9929   0.9370   0.9360   0.9690   0.9779
## Specificity            0.9922   0.9862   0.9859   0.9964   0.9963
## Pos Pred Value         0.9806   0.9420   0.9333   0.9812   0.9833
## Neg Pred Value         0.9971   0.9849   0.9865   0.9939   0.9950
## Prevalence             0.2843   0.1934   0.1746   0.1640   0.1838
## Detection Rate         0.2822   0.1812   0.1635   0.1589   0.1797
## Detection Prevalence   0.2878   0.1924   0.1751   0.1619   0.1827
## Balanced Accuracy      0.9925   0.9616   0.9610   0.9827   0.9871
# Run against 20 testing set provided by Professor Leek.
print(predict(modFit, newdata=df_testing))
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
# Train on training set 4 of 4 with only cross validation.
set.seed(666)
modFit <- train(df_small_training4$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=df_small_training4)
print(modFit, digits=3)
## Random Forest 
## 
## 2958 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2218, 2219, 2219, 2218 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.95      0.937  0.00656      0.00834 
##   27    0.955     0.943  0.00891      0.0113  
##   52    0.947     0.932  0.0101       0.0128  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
# Run against testing set 4 of 4.
predictions <- predict(modFit, newdata=df_small_testing4)
print(confusionMatrix(predictions, df_small_testing4$classe), digits=4)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 553  20   0   0   0
##          B   4 357  19   3   3
##          C   2   4 315   7   7
##          D   1   0   9 312   6
##          E   0   0   0   1 346
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9563          
##                  95% CI : (0.9463, 0.9649)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9447          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9875   0.9370   0.9184   0.9659   0.9558
## Specificity            0.9858   0.9817   0.9877   0.9903   0.9994
## Pos Pred Value         0.9651   0.9249   0.9403   0.9512   0.9971
## Neg Pred Value         0.9950   0.9848   0.9829   0.9933   0.9901
## Prevalence             0.2844   0.1935   0.1742   0.1640   0.1838
## Detection Rate         0.2809   0.1813   0.1600   0.1585   0.1757
## Detection Prevalence   0.2910   0.1960   0.1701   0.1666   0.1762
## Balanced Accuracy      0.9867   0.9594   0.9530   0.9781   0.9776
# Run against 20 testing set provided by Professor Leek.
print(predict(modFit, newdata=df_testing))
##  [1] B A B A A E D D A A B C B A E E A B B B
## Levels: A B C D E
Out of Sample Error

According to Professor Leek's Week 1 “In and out of sample errors”, the out of sample error is the “error rate you get on new data set.” In my case, it's the error rate after running the predict() function on the 4 testing sets:

Random Forest (preprocessing and cross validation) Testing Set 1: 1 - .9714 = 0.0286
Random Forest (preprocessing and cross validation) Testing Set 2: 1 - .9634 = 0.0366
Random Forest (preprocessing and cross validation) Testing Set 3: 1 - .9655 = 0.0345
Random Forest (preprocessing and cross validation) Testing Set 4: 1 - .9563 = 0.0437
Since each testing set is roughly of equal size, I decided to average the out of sample error rates derived by applying the random forest method with both preprocessing and cross validation against test sets 1-4 yielding a predicted out of sample rate of 0.03585.

CONCLUSION

I received three separate predictions by appling the 4 models against the actual 20 item training set:

A) Accuracy Rate 0.0286 Predictions: B A A A A E D B A A B C B A E E A B B B

B) Accuracy Rates 0.0366 and 0.0345 Predictions: B A B A A E D B A A B C B A E E A B B B

C) Accuracy Rate 0.0437 Predictions: B A B A A E D D A A B C B A E E A B B B

Since Professor Leek is allowing 2 submissions for each problem, I decided to attempt with the two most likely prediction sets: option A and option B.

Since options A and B above only differed for item 3 (A for option A, B for option B), I subimitted one value for problems 1-2 and 4-20, while I submitted two values for problem 3. For problem 3, I was expecting the automated grader to tell me which answer (A or B) was correct, but instead the grader simply told me I had a correct answer. All other answers were also correct, resulting in a score of 100%.
