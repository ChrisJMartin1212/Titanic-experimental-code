setwd("C:/Users/cjmartin/Desktop/Data Science Studies/Kaggle Titanic")
#install pkgs
install.packages("ggplot2")
install.packages("dplyr")
install.packages("caret")
install.packages("ranger")
install.packages("randomForest")
install.packages("tidyverse")
install.packages("rpart")
install.packages("party")
install.packages("rpart")


#read in pkgs
library(dplyr)
library(caret)
library(ranger)
library(tidyverse)
library(e1071)
library(party)
library(rpart.plot)
library(ggplot2)
#load csv files
Titanic_train <- read.csv("traincleanedchgd.csv", stringsAsFactors = F)
head(Titanic_train)
Titanic_test <- read.csv("testcleanedchgd.csv",stringsAsFactors = F)
head(Titanic_test)
full_Titanic<-rbind(Titanic_train, Titanic_test) 
str(full_Titanic)
#IGNORE?
preds <- Titanic_train %>% select(-Survived) %>% as.matrix()
output <- Titanic_train$Survived

#set up decision tree
Tpreds <- Titanic_train %>% select(-Survived) %>% as.matrix()
Tittree <- ctree(Survived ~ PassengerId + Sex + Pclass + Age + SibSp + Parch + Embarked, data = Titanic_train)
plot(Tittree)
# predict survived and write to file
letsee <- round(predict(Tittree, Titanic_test))
write.csv(letsee, file = "my2ndtree.csv")
#plotting bar graphs w/Survived and variables
ggplot(data = Tit_train, aes(x = Embarked)) + geom_bar(aes(fill = "red")) + facet_wrap(~Survived)
	

Tit_train$Sex <- Tit_train$Sex * 10
head(Tit_train)
*linear regression modeling
MODEL <- glm(Survived ~., family=binomial(link=logit), data=Titanic_train)
anova(MODEL, test="Chisq")

MODEL2 <- glm(Survived ~ Pclass + Sex + Age, family=binomial(link=logit), data=Titanic_train)
#predicting with linear regression
pred <- predict(MODEL2, newdata = Titanic_test, type = "response")

y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
y_act <- Titanic_train$Survived
mean(y_pred == y_act)

write.csv(y_pred, file = "mythirdtree.csv")

----------------
#after combining Titanic_train and Titanic_test into full_Titanic via rbind, we start plotting variables

# embarked vs. Pclass vs. Survived (and other combos)
ggplot(full_Titanic[1:891,],aes(x = Embarked, fill=factor(Survived))) +
geom_bar() + 
facet_wrap(~Pclass) + 
ggtitle("Pclass vs Embarked vs Survival")+
xlab("Embarked") +
ylab("Total Count") +
labs(fill = "Survived")  

# Family size created bby adding 1, SibSp, & Parch; then separating into three classes
#and adding a three-class factor column too
full_Titanic$FamilySize <- full_Titanic$SibSp + full_Titanic$Parch + 1

full_Titanic$FamilySized[full_Titanic$FamilySize == 1]   <- 'Single'
full_Titanic$FamilySized[full_Titanic$FamilySize < 5 & full_Titanic$FamilySize >= 2]   <- 'Small'
full_Titanic$FamilySized[full_Titanic$FamilySize >= 5]   <- 'Big'
full_Titanic$FamilySized=as.factor(full_Titanic$FamilySized)

#crossvalidation- setting aside 10% of training data
set.seed(333)
ind=createDataPartition(Titanic_train$Survived,times=1,p=0.9,list=FALSE)
train_val=Titanic_train[ind,]
partition_val=Titanic_train[-ind,]
#check the proportion of Survival rate in Titanic_train, train_val$Survived, & partition_val$Survived
round(prop.table(table(Titanic_train$Survived)*100),digits = 1)
round(prop.table(table(train_val$Survived)*100),digits = 1)
round(prop.table(table(partition_val$Survived)*100),digits = 1)
#plotting a decision tree
set.seed(198)
Model_DT=rpart(Survived~.,data=train_val,method="class")
rpart.plot(Model_DT,extra =  3,fallen.leaves = T)
#check its accuracy
PRE_TDT=predict(Model_DT,data=train_val,type="class")
confusionMatrix(PRE_TDT,train_val$Survived)
#crossvalidation of decision tree [this didn't work]
set.seed(198)
cv.10 <- createMultiFolds(train_val$Survived, k = 10, times = 10)
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                       index = cv.10)
Model_CDT <- train(x = train_val[,-7], y = train_val[,7], method = "rpart", tuneLength = 30,
                   trControl = ctrl)
#Random forest
set.seed(198)
RF1 <- randomForest(x = train_val[,-5],y=train_val[,5], importance = TRUE, ntree = 1000)
##crossvalidation of random forest
set.seed(18)
cv10_1 <- createMultiFolds(train_val[,5], k = 10, times = 10)
#using caret's train control
ctrl_1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                      index = cv10_1)
set.seed(198)
rf.5<- train(x = train_val[,-5], y = train_val[,5], method = "rf", tuneLength = 3,
              ntree = 1000, trControl =ctrl_1)
#predicting test data
pr.rf=predict(rf.5,newdata = partition_val)

confusionMatrix(pr.rf,partition_val$Survived)
#non-linear SVM, radial kernel ???
set.seed(127)

rd.poly=tune.svm(Survived~.,data=train_val,kernel="radial",gamma=seq(0.1,5))

summary(rd.poly)
#predicting
pre.rd=predict(best.rd,newdata = partition_val)
confusionMatrix(pre.rd,partition_val$Survived)
#logistic regression
log.mod <- glm(Survived ~ ., family = binomial(link=logit), 
               data =train_val)

test.probs <- predict(log.mod, newdata=partition_val,type =  "response")
table(partition_val$Survived,test.probs>0.5)