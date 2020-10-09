######################### EMPLOYEE CAR USAGE ANALYSIS #####################
### Setting up the working directory ####
setwd("C:/greatlakes/ML/project")
getwd()
### Invoking the necessary libraries ####
install.packages("readr")
library(readr)
install.packages("Hmisc")
library(Hmisc)
install.packages("ggplot2")
library(ggplot2)
install.packages("psych")
library(psych)
install.packages("caret")
library(caret)
install.packages("class")
library(class)
install.packages("caTools")
library(caTools)
install.packages("e1071")
library(e1071)
install.packages("DMwR")
library(DMwR)
install.packages("xgboost")
library(xgboost)
install.packages("gbm")
library(gbm)
install.packages("Matrix")
library(Matrix)
install.packages("ipred")
library(ipred)
install.packages("rpart")
library(rpart)
### Importing the dataset ####
trans = read.csv("Cars.csv",header = TRUE)
### Identification of different variables ####
View(trans)
dim(trans)
str(trans)
head(trans)
tail(trans)
summary(trans)

### Conversion of cateogrical variable ###
trans$Gender = as.factor(trans$Gender)
trans$Gender = ifelse(trans$Gender == "Male","1","0")
trans$Gender = as.factor(trans$Gender)
summary(trans$Gender)
trans$license = as.factor(trans$license)
trans$Transport = ifelse(trans$Transport == "Car","1","0")
class(trans$Transport)
trans$Transport = as.factor(trans$Transport)
trans$Engineer = as.factor(trans$Engineer)
trans$MBA = as.factor(trans$MBA)
### Uni-Variate Analysis ####
## Analysis of independent numerical variable ##
cor = trans[,-c(2,3,4,8,9)]
hist.data.frame(cor,freq = TRUE)
table(trans[,c(2,3,4,8,9)])
## Analysis of independent categorical variable ##
# Engineer #
qplot(trans$Engineer,fill = trans$Engineer,main = "Engineer")
prop.table(table(trans$Engineer))
# Gender #
qplot(trans$Gender,fill = trans$Gender,main = "Gender")
prop.table(table(trans$Gender))
# MBA #
qplot(trans$MBA,fill = trans$MBA,main = "MBA")
prop.table(table(trans$MBA))
# License #
qplot(trans$license,fill = trans$license, main = "license")
prop.table(table(trans$license))
## Analysis of dependent categorical variable
# Transport #
qplot(trans$Transport,fill = trans$Transport, main = "Transport")
table(trans$Transport)
prop.table(table(trans$Transport))

### Bi-Variate Analysis ####
## Analysis of dependent variable with numerical variables ##
qplot(Distance,fill = Transport,data = trans,
      main = "Transport and Distance")
qplot(Age,fill = Transport,data = trans,
      main = "Transport and Age",geom = "bar")
qplot(Salary,fill = Transport,data = trans,
      main = "Transport and Salary",
      geom = "density") 
qplot(Work.Exp,fill = Transport,data = trans,
      main = "Transport and Work.Exp",
      geom = "dotplot")
## Analysis of dependent variable with categorical variables ##
qplot(trans$Transport,fill = trans$Gender,xlab = "Transport",
      ylab = "Gender",main = "Gender vs. Transport")
qplot(trans$Transport,fill = trans$MBA,xlab = "Transport",
      ylab = "MBA",main = "Transport vs. MBA")
qplot(trans$Transport,fill = trans$license,xlab = "Transport",
      ylab = "license",main = "Transport vs. license")
## Analysis of independent variables with independent variables ##
qplot(Age,Salary, fill = Gender,data = trans,
      geom = "bin2d",main = "Age vs. Salary")
qplot(Salary,fill = MBA,data = trans,geom = "density",
      main = "Salary vs. MBA")
qplot(Age,Distance,data = trans,fill = license,geom = "polygon",
      main = "Age vs. Distance")
qplot(Work.Exp,Salary,fill = Engineer,data = trans,
      geom = "boxplot",main = "Work.Exp vs. Salary")
### Missing values treatment ####
sum(is.na(trans))
trans[is.na(trans)] = 1
sum(is.na(trans))
###

### Outlier Treatment ####
boxplot(trans[,-c(2,3,4,8,9)])


### Multi collinearity #####
## Correlation Plot ###
cor = trans[,-c(2,3,4,8,9)]
cor.plot(cor,numbers = TRUE)
## Eigen Values ###
eigen = eigen(cor(trans[,c(1,5,6,7)]))
eigen$values
## Scatter Plot ###
plot(cor)


### Building the models ###
### Splitting of the datasets ###
set.seed(77)
splt = sample.split(trans$Transport,SplitRatio = 0.7)
m.train = subset(trans,splt == TRUE)
m.test = subset(trans,splt == FALSE)
dim(m.train)
dim(m.test)
summary(m.train$Transport)
### Building the logistic regression model ####
set.seed(77)
mglm = glm(Transport~.,m.train,family = "binomial",maxit = 100)
summary(mglm)
## Predictions for Logistic Regression ###
set.seed(77)
m.test.pred = predict(mglm,m.test,type = "response")
m.test.predc = ifelse(m.test.pred > 0.5,"1","0")
m.test.predc = as.factor(m.test.predc)
m.test$Transport = as.factor(m.test$Transport)
## Confusion Matrix ###
cf.lr = caret::confusionMatrix(m.test$Transport,m.test.predc,positive = "1")
print(cf.lr)
cf.lr$byClass
plot(m.test$Transport,m.test.predc,xlab = "Actuals",ylab = "Predicted",
     col = c("Blue","Red"))

### Building a Naive Bayes Model ####
set.seed(77)
m.nb = naiveBayes(m.train$Transport~.,data = m.train)
print(m.nb)
## Predictions for Naive Bayes ###
set.seed(77)
m.nb.test.pred = predict(m.nb,m.test,type = "class")
## Confusion Matrix ###
nb.cf = caret::confusionMatrix(m.test$Transport,m.nb.test.pred,positive = "1")
print(nb.cf)
nb.cf$byClass
plot(m.test$Transport,m.nb.test.pred,xlab = "Actuals",ylab = "Predicted",
     col = c("Purple","Pink"))

### Building a KNN model #### 
set.seed(77)
m.knn = knn(train = m.train,test = m.test,cl = m.train$Transport,k = 17)
## Confusion Matrix ###
knn.cf = caret::confusionMatrix(m.test$Transport,m.knn,positive = "1")
print(knn.cf)
knn.cf$byClass
plot(m.test$Transport,m.knn,xlab = "Actuals",ylab = "Predicted",
     col = c("Brown","Yellow"))

### Doing a SMOTE analysis for a better dataset ####
set.seed(77)
s.train =SMOTE(Transport~.,m.train,perc.over = 501, perc.under = 121)
s.test = m.test
prop.table(table(s.train$Transport))
dim(s.train)

### Building a Logistic Regression Model after SMOTE ####
set.seed(77)
sglm = glm(Transport~.,data = s.train,family = "binomial",maxit = 100)
summary(sglm)
s.train$Gender
## Predictions for Logistic Regression after SMOTE ###
set.seed(77)
s.test.pred = predict(sglm,s.test,type = "response")
s.test.predc = ifelse(s.test.pred > 0.5,"1","0")
s.test.predc = as.factor(s.test.predc)
s.test$Transport = as.factor(s.test$Transport)
## Confusion Matrix ###
glm.c = caret::confusionMatrix(s.test$Transport,s.test.predc,positive = "1")
print(glm.c)
glm.c$byClass
plot(s.test$Transport,s.test.predc,xlab = "Actuals",ylab = "Predicted",
     col = c("Green","Violet"))

### Building a Naive Bayes Model after SMOTE####
set.seed(77)
s.nb = naiveBayes(s.train$Transport~.,data = s.train)
print(s.nb)
## Predictions for Naive Bayes after SMOTE ###
s.nb.test.pred = predict(s.nb,s.test,type = "class")
## Confusion Matrix ###+
nb.c = caret::confusionMatrix(s.test$Transport,s.nb.test.pred,positive = "1")
print(nb.c)
nb.c$byClass
plot(s.test$Transport,s.nb.test.pred,xlab = "Actuals",ylab = "Predicted",
     col = c("Black","Orange"))
### Building a KNN model after SMOTE####
set.seed(77)
s.knn = knn(train = s.train,test = s.test,cl = s.train$Transport,k = 17)
### Confusion Matrix ##
knn.c = caret::confusionMatrix(s.test$Transport,s.knn,positive = "1")
print(knn.c)
knn.c$byClass
plot(s.test$Transport,s.knn,xlab = "Actuals",ylab = "Predicted",
     col = c("Black","Gray"))

### BAGGING AFTER SMOTE ####
set.seed(77)
bag.model = bagging(Transport~.,data = s.train,coob = TRUE,
                    control = rpart.control(maxdepth = 10,minsplit = 3))
summary(bag.model)
## Making predictions with Bagging model ####
set.seed(77)
s.bag.pred = predict(bag.model,s.test,type = "class")
bag.pred = as.factor(s.bag.pred)
## Confusion Matrix
bag.c = caret::confusionMatrix(s.test$Transport,bag.pred,positive = "1")
print(bag.c)
bag.c$byClass
plot(s.test$Transport,bag.pred,xlab = "Actuals",ylab = "Predicted",
     col = c("White","Red"))
### Boosting after SMOTE ####
### Ada boost ####
s.test$Transport = as.numeric(s.test$Transport)
str(s.train)
s.test$Transport = as.factor(s.test$Transport)
s.train$Transport = as.factor(s.train$Transport)
set.seed(77)
s.gbm <- gbm(
        formula = Transport~.,
        data = s.train,
        distribution = "multinomial",
        n.trees = 100,
        interaction.depth = 1,
        shrinkage = 0.1,
        cv.folds = 2,
        n.cores = 2, # will use all cores by default
        verbose = FALSE
)
summary(s.gbm)
## Predictions for Adaboost model ###
set.seed(77)
prob.s.gbm = predict(s.gbm,s.test,type = "response")
prob.s.gbm = as.data.frame(prob.s.gbm)
qplot(prob.s.gbm[,c(1)],s.test$Transport,geom = "boxplot")
pred.s.gbm = ifelse(prob.s.gbm[,c(1)] > 0.50,"0","1")
pred.s.gbm = as.factor(pred.s.gbm)
s.test = m.test
## Confusion Matrix ###
adb.c = caret::confusionMatrix(s.test$Transport,pred.s.gbm,positive = "1")
print(adb.c)
adb.c$byClass
plot(s.test$Transport,pred.s.gbm,xlab = "Actuals",ylab = "Predicted",
     col = c("Blue","Yellow"))
colnames(test)
colnames(train)
table(test$loan_status)

### XG BOOSTING ####
set.seed(77)
f = model.matrix(s.train$Transport~.,s.train)

t = s.train[,9]
t = as.matrix(t)
t = as.character(t)
t = as.numeric(t)
xg.train.m = xgb.DMatrix(data = f,label = t)

e = model.matrix(s.test$Transport~.,s.test)
s = s.test[,9]
s = as.matrix(s)
s = as.character(s)
s = as.numeric(s)
xg.test.m = xgb.DMatrix(data = e,label = s)
set.seed(77)
s.xgb <- xgboost(
  data = xg.train.m,
  eta = 0.5,
  max_depth = 5,
  nrounds = 2,
  nfold = 2,
  objective = "binary:logistic",  # for regression models
  verbose = 0,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)
summary(s.xgb)
s.xgb$nfeatures
s.xgb$feature_names
xg.test.m
## Predictions for XGboost ###
xgb.predict = predict(s.xgb,xg.test.m)
bp = qplot(xgb.predict,s.test$Transport,geom = "boxplot"
      ,xlab = "Probabilities",ylab = "Transport",xintercept = 0.63)
bp+geom_vline(xintercept = 0.6126,color = "Red",size = 1.5)
xgb.predict.c = ifelse(xgb.predict > 0.6126,"1","0")
xgb.predict.c = as.factor(xgb.predict.c)
## Confusion Matrix ###xgb.c = caret::confusionMatrix(s.test$Transport,xgb.predict.c,positive = "1")

print(xgb.c)
xgb.c$byClass
plot(s.test$Transport,xgb.predict.c,xlab = "Actuals",ylab = "Predicted",
     col = c("Yellow","Red"))

