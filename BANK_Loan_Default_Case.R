rm(list = ls(all=T))
Bank_loan = read.csv("F:/PROJECTs/Bank_Loan_Default_Case/bank-loan.csv")
View(Bank_loan)
str(Bank_loan)

##Conversion the variables into required type
Bank_loan$age = as.numeric(Bank_loan$age)
Bank_loan$ed = as.factor(Bank_loan$ed)
Bank_loan$employ = as.numeric(Bank_loan$employ)
Bank_loan$address = as.numeric(Bank_loan$address)
Bank_loan$income = as.numeric(Bank_loan$income)
Bank_loan$default = as.factor(Bank_loan$default)

##Check the conversion
str(Bank_loan)

##Distribution of data
hist(Bank_loan$age)
hist(Bank_loan$employ)
hist(Bank_loan$address)
hist(Bank_loan$income)
hist(Bank_loan$debtinc)
hist(Bank_loan$creddebt)
hist(Bank_loan$othdebt)

#plot relation with each x and Y

attach(Bank_loan)
plot.default(ed)
plot.default(age)
plot.default(address)
plot.default(income)
plot.default(debtinc)
plot.default(creddebt)
plot.default(othdebt)



## Check for the missing value
sum(is.na(Bank_loan))

##Check the missing value in which variable
sum(is.na(Bank_loan$age))
sum(is.na(Bank_loan$ed))
sum(is.na(Bank_loan$employ))
sum(is.na(Bank_loan$address))
sum(is.na(Bank_loan$income))
sum(is.na(Bank_loan$debtinc))
sum(is.na(Bank_loan$creddebt))
sum(is.na(Bank_loan$othdebt))
sum(is.na(Bank_loan$default))
sum(is.na(Bank_loan))

##We can remove the na value or we can also impute the na value(But we can not simply remove the na value because in logistic regression we need more data so imputing is best option)
##All the missing value lies only in default variable and default variable is factor so we have to impute with mode.
##So for finding the mode of default variable we have to use the table function

table(Bank_loan$default)

Bank_loan$default[is.na(Bank_loan$default)] = 0
sum(is.na(Bank_loan$default))

##For feature selection
install.packages("Boruta")

library(Boruta)
set.seed(111)
boruta = Boruta(default ~., data = Bank_loan, doTrace = 2)
plot(boruta)
plotImpHistory(boruta)
attStats(boruta)
# Scatter plot of all plots with all variables

pairs(Bank_loan)

#we can also see correlation coefficient and scatter plot together
install.packages("GGally")
install.packages("stringi")
library(GGally)
library(stringi)
windows()
ggpairs(Bank_loan)



##Check for the outliers
boxplot(Bank_loan$age)

table(Bank_loan$employ)
boxplot(Bank_loan$employ)
boxplot(Bank_loan$employ)$out

boxplot(Bank_loan$address)
boxplot(address)$out


boxplot(Bank_loan$income)
boxplot(Bank_loan$income)$out

boxplot(Bank_loan$debtinc)
boxplot(Bank_loan$debtinc)$out

boxplot(Bank_loan$creddebt)
boxplot(Bank_loan$creddebt)$out

boxplot(Bank_loan$othdebt)
boxplot(Bank_loan$othdebt)$out

##Split the data into train and test

library(caTools)
split = sample.split(Bank_loan, SplitRatio = 0.70)

split
train_data = subset(Bank_loan, split==TRUE)
test_data = subset(Bank_loan, split==FALSE)



##General logistic model

model1 = glm(default~., family = "binomial", data = train_data)
summary(model1)

model2 = glm(default~.-ed, family = "binomial", data = train_data)
summary(model2)

model3 = glm(default~.-age,family = "binomial", data = train_data)
summary(model3)

model4 = glm(default~.-income,family = "binomial", data = train_data)
summary(model4)

model5 = glm(default~.-othdebt,family = "binomial", data = train_data)
summary(model5)

##Check the accuracy

prob = predict(model1,type = c("response"),train_data)
prob

confusion = table(prob>0.50,train_data$default)
confusion
##Model accuracy
Accuracy = sum(diag(confusion)/sum(confusion))
Accuracy ###0.8127208

##Now check the accuracy for test data 
prob1 = predict(model1,type = c("response"),test_data) 
prob1
confusion_1 = table(prob1>0.50,test_data$default)
confusion_1
Accuracy_1 = sum(diag(confusion_1)/sum(confusion_1))
Accuracy_1 ##0.8450704
##Check the threshold value to decrease the false positive rate 
library(ROCR)
ROCRPred = prediction(prob1,test_data$default)
ROCRPref = performance(ROCRPred,"tpr","fpr")

plot(ROCRPref, colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))

prob = predict(model1,type = c("response"),train_data)
prob
confusion = table(prob>0.44,train_data$default)
confusion
Accuracy = sum(diag(confusion)/sum(confusion))
Accuracy

##Calculate the AUC
library(pROC)
auc = performance(ROCRPred,measure = "auc")
auc = auc@y.values[[1]]
auc



##We use Decision tree algorithm for enhancing accuracy
prop.table(table(train_data$default))
prop.table(table(test_data$default))

##Library for Decision Tree
library(C50)

##Building model on Training Data
Train_model = C5.0(train_data[,-9],train_data$default)
plot(Train_model)

##Training Accuracy

pred_train = predict(Train_model, train_data[,-9])
table(pred_train, train_data$default)

mean(train_data$default==predict(Train_model,train_data)) ## 0.903177

##Testing Accuracy
pred_test = predict(Train_model,newdata = test_data[,-9])
table(pred_test, test_data$default)

mean(pred_test==test_data$default)



##By using Random Forest
library(randomForest)

#Building the Random Forest model on training model
fit.forest = randomForest(default~.,data = train_data, na.action = na.roughfix, importance=TRUE)

##Fit.forest(Prediction)
pred_t1 = fit.forest$predicted
table(pred_t1,train_data$default)
mean(train_data$default==pred_t1) ##0.7844523

##Predicting accuracy on test data
pred_t2 = predict(fit.forest,newdata = test_data[,-9])
table(pred_t2,test_data$default)
mean(test_data$default==pred_t2) ##0.8204225


##This model should be exceotable because the accuracy of both the tran and test data almost equal
##confusion matrix(using caret)
library(caret)
confusionMatrix(train_data$default,fit.forest$predicted)
confusionMatrix(test_data$default,pred_t2)

##Visualisation
plot(fit.forest,lwd=2)
legend("topright",colnames(fit.forest$err.rate),col = 1:4,cex = 0.8,fill = 1:4)

##Crosstable
library(gmodels)
rf_perf=CrossTable(train_data$default,fit.forest$predicted,prop.chisq = FALSE,prop.c = FALSE,prop.r = FALSE,dnn = c("actual default","predicted default"))




##Xgboosting
install.packages("xgboost")
library(xgboost)
library(magrittr)
library(Matrix)
library(dplyr)

id=sample(2,nrow(Bank_loan),prob = c(0.80,0.20),replace = TRUE)
B_train = Bank_loan[id==1,]
B_test = Bank_loan[id==2,]
str(B_train)
train_m = sparse.model.matrix(default~ ., data = train_data)
head(train_m)

train_label = train_data[,"default"]
train_matrix = xgb.DMatrix(data = as.matrix(train_m),label = train_label)

test_m = sparse.model.matrix(default~., data = test_data)
test_label = test_data[,"default"]
test_matrix = xgb.DMatrix(data = as.matrix(test_m),label = test_label)

##Parameters
nc = length(unique(train_label))
xgb__params = list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = 3)
watchlist = list(train = train_matrix, test = test_matrix)


##Gradient boosting model
best_model = xgb.train(params = xgb__params,
                       data = train_matrix,
                       nrounds = 1000,
                       watchlist = watchlist,
                       eta = 0.05,
                       max.depth = 3,
                       gamma = 0,
                       subsample = 1,
                       colsample_bytree = 1,
                       missing = NA,
                       seed = 333)


best_model

e = data.frame(best_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = "blue")

lines(e$iter, e$test_mlogloss, col = "red")
min(e$test_mlogloss)
e[e$test_mlogloss==0.478741,]

##Feature importance
imp = xgb.importance(colnames(train_matrix),model = best_model)
print(imp)
xgb.plot.importance(imp)

##Prediction and confusion matrix using test data

 
p = predict(best_model,newdata = test_matrix)
head(p)

length(test_matrix)
dim(test_matrix)
length(test_label)
dim(test_label)
preed = matrix(p, nrow = nc, ncol = length(p)/nc) %>%
  t() %>%
  data.frame() %>%
  mutate(label=test_label, max_prob = max.col(., "last")-1)

