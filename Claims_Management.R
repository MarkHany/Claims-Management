library(dplyr)
library(ggplot2)
library(data.table)
library(tidyr)
library(zoo)
library(caret)
library(xgboost)
library(class)
library(gmodels)

rm(list=ls())

df<- fread("train.csv")
testdf <- fread("D:/Mark/Study/GP/Claims Management/train.csv/test.csv/test.csv")
View(df)
str(df)
summary(df)

View(testdf)
str(testdf)
summary(testdf)

setDT(df)
#Preprocessing and Preparing Data
sapply(df,function(x)any(is.na(x)))

cat("Preprocess data\n")

#v1:v2
#v4:v21
#v23
#v25:v29
#v32:v37
#v39:v46
#v48:v51
#v53:v61
#v63:v65
#v67:v70
#v73
#v76:v78
#v80:v90
#v92:v106
#v108:v109
#v111
#v114:v124
#v125:v128
#v130:v131

###########################Cleaning the NA's############################


#Drop rows containing NA:
CleanTable <- na.omit(df)
summary(CleanTable)

#######################################################################

##Cleaning V3

#change "" into "Unknown"
CleanTable$v3[CleanTable$v3==""] <- "UNKNOWN"

#Change v3 into factor
CleanTable$v3 <- as.factor(CleanTable$v3)

##Cleaning v22
table(CleanTable$v22)

#change "" into "Unknown"
CleanTable$v22[CleanTable$v22==""] <- "UNKNOWN"

#Change v22 into factor
CleanTable$v22 <- as.factor(CleanTable$v22)

##Cleaning v24
table(CleanTable$v24)

#Change v24 into factor
CleanTable$v24 <- as.factor(CleanTable$v24)

##Cleaning v30
table(CleanTable$v30)

#change "" into "Unknown"
CleanTable$v30[CleanTable$v30==""] <- "UNKNOWN"

#Change v30 into factor
CleanTable$v30 <- as.factor(CleanTable$v30)

##Cleaning v31
table(CleanTable$v31)

#change "" into "Unknown"
CleanTable$v31[CleanTable$v31==""] <- "UNKNOWN"

#Change v31 into factor
CleanTable$v31 <- as.factor(CleanTable$v31)


##Cleaning v38
table(CleanTable$v38)

#Change v38 into factor
CleanTable$v38 <- as.factor(CleanTable$v38)


##Cleaning v47
table(CleanTable$v47)

#Change v47 into factor
CleanTable$v47 <- as.factor(CleanTable$v47)


##Cleaning v52
table(CleanTable$v52)

#Change v52 into factor
CleanTable$v52 <- as.factor(CleanTable$v52)


##Cleaning v56
table(CleanTable$v56)

#change "" into "Unknown"
CleanTable$v56[CleanTable$v56==""] <- "UNKNOWN"

#Change v56 into factor
CleanTable$v56 <- as.factor(CleanTable$v56)

##Cleaning v62
table(CleanTable$v62)

#Change v62 into factor
CleanTable$v62 <- as.factor(CleanTable$v62)

##Cleaning v66
table(CleanTable$v66)

#Change v66 into factor
CleanTable$v66 <- as.factor(CleanTable$v66)

##Cleaning v71
table(CleanTable$v71)

#Change v71 into factor
CleanTable$v71 <- as.factor(CleanTable$v71)

##Cleaning v72
table(CleanTable$v72)

#Change v72 into factor
CleanTable$v72 <- as.factor(CleanTable$v72)

##Cleaning v74
table(CleanTable$v74)

#Change v74 into factor
CleanTable$v74 <- as.factor(CleanTable$v74)

##Cleaning v75
table(CleanTable$v75)

#Change v75 into factor
CleanTable$v75 <- as.factor(CleanTable$v75)

##Cleaning v79
table(CleanTable$v79)

#Change v79 into factor
CleanTable$v79 <- as.factor(CleanTable$v79)

##Cleaning v91
table(CleanTable$v91)

#Change v91 into factor
CleanTable$v91 <- as.factor(CleanTable$v91)

##Cleaning v107
table(CleanTable$v107)

#Change v107 into factor
CleanTable$v107 <- as.factor(CleanTable$v107)

##Cleaning v110
table(CleanTable$v110)

#Change v110 into factor
CleanTable$v110 <- as.factor(CleanTable$v110)

##Cleaning v112
table(CleanTable$v112)

#change "" into "Unknown"
CleanTable$v112[CleanTable$v112==""] <- "UNKNOWN"

#Change v112 into factor
CleanTable$v112 <- as.factor(CleanTable$v112)

##Cleaning v113
table(CleanTable$v113)

#change "" into "Unknown"
CleanTable$v113[CleanTable$v113==""] <- "UNKNOWN"

#Change v113 into factor
CleanTable$v113 <- as.factor(CleanTable$v113)

##Cleaning v125
table(CleanTable$v125)

#change "" into "Unknown"
CleanTable$v125[CleanTable$v125==""] <- "UNKNOWN"

#Change v125 into factor
CleanTable$v125 <- as.factor(CleanTable$v125)


sapply(CleanTable,function(x)any(is.na(x)))

sapply(CleanTable,function(x)any(is.character(x)))

sapply(CleanTable,function(x)any(is.numeric(x)))

sapply(CleanTable,function(x)any(is.factor(x)))

table(CleanTable$target)

#Preprocess the Target and change it into True/False
CleanTable$target <- as.logical(CleanTable$target)

#Check if there is any other variables need to be cleaned
summary(CleanTable)


#Check for Correlation
CleanTable[,3]

CleanTable[,133]

nums <- unlist(lapply(CleanTable, is.numeric))  

CleanTable[,nums]

cor.mat <- cor(CleanTable[,nums])
cor.mat

hc = findCorrelation(cor.mat, cutoff=0.75)

hc = sort(hc)

reduced_Data = CleanTable[,-c(hc)]
print(reduced_Data)

nums <- unlist(lapply(reduced_Data, is.numeric))  

cor.mat <- cor(reduced_Data[,nums])
cor.mat

hc2 = findCorrelation(cor.mat, cutoff=0.75)

hc2 = sort(hc2)
reduced_Data2 = reduced_Data[,-c(hc)]

nums <- unlist(lapply(reduced_Data2, is.numeric))  

cor.mat2 <- cor(reduced_Data2[,nums])
cor.mat2


################################
#####Data Visualization####################

v3count <- table(CleanTable$target, CleanTable$v3) 
v24count <- table(CleanTable$target, CleanTable$v24)
v30count <- table(CleanTable$target, CleanTable$v30)
v38count <- table(CleanTable$target, CleanTable$v38)
v47count <- table(CleanTable$target, CleanTable$v47)
v52count <- table(CleanTable$target, CleanTable$v52)
v62count <- table(CleanTable$target, CleanTable$v62)
v66count <- table(CleanTable$target, CleanTable$v66)
v71count <- table(CleanTable$target, CleanTable$v71)
v72count <- table(CleanTable$target, CleanTable$v72)
v74count <- table(CleanTable$target, CleanTable$v74)
v75count <- table(CleanTable$target, CleanTable$v75)
v91count <- table(CleanTable$target, CleanTable$v91)
v107count <- table(CleanTable$target, CleanTable$v107)
v110count <- table(CleanTable$target, CleanTable$v110)
v129count <- table(CleanTable$target, CleanTable$v129)

#barplots of frequencies
par(mfrow=c(2,2))
barplot(v3count, ylab="#", main="v3count", legend.text=TRUE)
barplot(v24count, ylab="#", main="v24count", legend.text=TRUE)
barplot(v30count, ylab="#", main="v30count", legend.text=TRUE)
barplot(v38count, ylab="#", main="v38count", legend.text=TRUE)



#From the four plots above I can see:
#:: The v3 variable is a strong predictor for the target of 1 when the variable is equal to C

#:: The same can be said for the v38 variable, it seems to be a strong predictor of a target of 1 when the variable is equal to 0

#:: The responses for the v24 variable is mostly 'E' with a large proportion of them corresponding to a target value of 1. It's not quite as strong (doesn't stand out as much graphically) as the v3 and v38 variables, though.

#:: The v30 variable looks slightly bimodal, but heavily favors 'C' as a predictor of a target value of 1. However, the counts are extremely low.

#barplots of frequencies
par(mfrow=c(2,2))
barplot(v47count, ylab="#", main="v47count", legend.text=TRUE)
barplot(v52count, ylab="#", main="v52count", legend.text=TRUE)
barplot(v62count, ylab="#", main="v62count", legend.text=TRUE)
barplot(v66count, ylab="#", main="v66count", legend.text=TRUE)

#From the four plots above I can see:
#:: The v47 variable is strongly bimodal. The 'C' and 'I' responses could possibly both be used as predictors of the target 1, however the 'C' response has a much higher proportion of 1 to 0 counts.

#:: The v52 variable doesn't look to be anything special, but it does tell me that -- no matter what the response for the v52 variable is -- it won't tell me much. This is because no one response stands out more than the others (e.g. higher frequency count) nor does one have a higher ratio of 1 to 0 target responses.

#:: The v62 variable is pretty clear that a response of 1 for v62 probably corresponds to a target value of 1.

#:: The v66 variable is also pretty clear that a response of A probably corresponds to a target value of 1 as well.


par(mfrow=c(2,2))
barplot(v71count, ylab="#", main="v71count", legend.text=TRUE)
barplot(v72count, ylab="#", main="v72count", legend.text=TRUE)
barplot(v74count, ylab="#", main="v74count", legend.text=TRUE)
barplot(v75count, ylab="#", main="v75count", legend.text=TRUE)





par(mfrow=c(2,2))
barplot(v91count, ylab="#", main="v91count", legend.text=TRUE)
barplot(v107count, ylab="#", main="v107count", legend.text=TRUE)
barplot(v110count, ylab="#", main="v110count", legend.text=TRUE)
barplot(v129count, ylab="#", main="v129count", legend.text=TRUE)




####################################


names(CleanTable)
HighCorrCols <- findCorrelation(cor.mat,cutoff = 0.70, verbose = TRUE,names = TRUE)

CleanTable2 <- CleanTable[,-which(names(CleanTable) %in% HighCorrCols)]



######################################

CategoricalName <- lapply(CleanTable2, is.factor)
CategoricalName <- names(CategoricalName[CategoricalName == TRUE])
  
for (f in names(CleanTable2)) {
  if (class(CleanTable2[[f]])=="character" || class(CleanTable2[[f]])=="factor") {
    CleanTable2[[f]] <- as.integer(factor(CleanTable2[[f]]))
  }
}


cor.mat3 <- cor(CleanTable2[,CategoricalName])
cor.mat3

HighCorrCatCols <- findCorrelation(cor.mat3,cutoff = 0.70, verbose = TRUE,names = TRUE)
CleanTable2 <- CleanTable2[,-which(names(CleanTable2) %in% HighCorrCatCols)]

##############Train the model###############

#### Logistic Regression Using XGboost ##############

xgtrain = xgb.DMatrix(as.matrix(CleanTable2), label = CleanTable2$target)

docv <- function(param0, iter) {
  model_cv = xgb.cv(
    params = param0
    , nrounds = iter
    , nfold = 2
    , data = xgtrain
    , early_stopping_rounds = 10
    , maximize = FALSE
    , nthread = 8
    , prediction = TRUE
    , verbose = TRUE
  )
  gc()
  best <- min(model_cv$test.auc.mean)
  bestIter <- which(model_cv$test.auc.mean==best)
  
  cat("\n",best, bestIter,"\n")
  print(model_cv[bestIter])
  
  bestIter-1
}


param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic"
  , "eval_metric" = "auc"
  , "eta" = 0.07
  , "subsample" = 0.9
  , "colsample_bytree" = 0.9
  , "min_child_weight" = 1
  , "max_depth" = 10
)

cat("Training a XGBoost classifier with cross-validation\n")
set.seed(2018)
cv <- xgb.cv(data = xgtrain, nrounds = 200, nthread = 8, nfold = 5, metrics = list("logloss"),
             max_depth = 10, eta = 0.07, objective = "binary:logistic")
cv


########## KNN ############

cat("Training using KNN with cross-validation\n")

TrainData <- CleanTable2[1:40664,-2]
TestData <- CleanTable2[40665:62561,-2]


TrainDataLabels <- CleanTable2[1:40664,2]
TestDataLabels <- CleanTable2[40665:62561,2]

KNN_CrossV <- knn.cv(train = TrainData,cl = TrainDataLabels,k=11)

KNN_Test_Pred <- knn(train = TrainData,test = TestData,cl = TrainDataLabels, k=2)

CrossTable(x=TrainDataLabels,y=KNN_CrossV,prop.chisq = FALSE)

accuracy = ((1031+27489)/40664)*100

########### Decision Trees ###############
library(rpart)
library(rpart.plot)
library(ROCR)

DT_TrainData <- CleanTable2[1:40664,-1]
DT_TestData <- CleanTable2[40665:62561,]



fit <- rpart(target ~ .,
             method="class", 
             data=DT_TrainData,
             control=rpart.control(minsplit=2, maxdepth = 3),
             parms=list(split='information'))


rpart.plot(fit, type = 4, extra = "auto")
summary(fit)

DT_pred <- predict(fit,DT_TestData,type = "class")

DT_confMat <- table(DT_TestData$target,DT_pred)

CrossTable(x=DT_TestData$target,y=DT_pred,prop.chisq = FALSE)

DT_accuracy <- sum(diag(DT_confMat))/sum(DT_confMat)

########## Naive Bayes ##########
library(e1071)

NB_TrainData <- CleanTable2[1:40664,-1]
NB_TestData <- CleanTable2[40665:62561,]


NB_model <- naiveBayes(target~.,NB_TrainData)

NB_model

NB_results <- predict (NB_model, NB_TestData)

NB_results

NB_confMat <- table(NB_TestData$target,NB_results)

CrossTable(x=NB_TestData$target,y=NB_results,prop.chisq = FALSE)

NB_accuracy <- sum(diag(NB_confMat))/sum(NB_confMat)

############## Neural Networks ###########
library(neuralnet)

NN_TrainData <- CleanTable2[1:40664,-1]
NN_TestData <- CleanTable2[40665:62561,]

NN_Labels <- names(NN_TrainData)

f <- as.formula(paste("target ~", paste(NN_Labels[!NN_Labels %in% "target"], collapse = " + ")))

print(net <- neuralnet(f,NN_TrainData, hidden=c(3), linear.output=FALSE, likelihood = TRUE))

plot(net)

NN_pred <- compute(net,NN_TestData[2:62])



######### Logistic Regression ############
