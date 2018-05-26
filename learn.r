library(caret)
library(ggplot2)
library(e1071)
library(ellipse)
library(lattice)

setwd("~/../Dropbox/Projects/personal_projects/machineLearning")

input_file <- "new_data.csv"

input_data <- read.table(input_file, header = FALSE)

# Create a dump data log
input_data.org = input_data


# Sets up the column names
colnames(input_data) <- c("Name","Credit_Score", "Weekly_Income","Other","Commute","Food","Rent")

# Ratio
input_data[,8] = (input_data.org[,4] + input_data.org[,5] + input_data.org[,6] + input_data.org[,7])/input_data.org[,3]

# Calc new input
input_data[,3] = input_data.org[,3] - (input_data.org[,5] + input_data.org[,7])

# reset out dump data to real data
input_data.org = input_data

# Divides by pre-rent numbers
input_data[,4] = input_data.org[,4]/input_data.org[,3]
input_data[,6] = input_data.org[,6]/input_data.org[,3]

colnames(input_data) <- c("Name","Credit_Score", "Weekly_Income","Other","Commute","Food","Rent", "Ratio")

set.seed(123123)

validation_index <- createDataPartition(input_data$Ratio, p=0.80, list=FALSE)
validation <- input_data[-validation_index,]
input_data <- input_data[validation_index,]


folds <- createMultiFolds(input_data$Name, k=10, times=20)
control <- trainControl(method="repeatedcv", number= 10, index = folds, repeats = 20)
control.ld <- trainControl(method = "cv", number =10)
metric <- "Accuracy"

library(doSNOW)

start.time <- Sys.time()

cl <- makeCluster(5,type = "SOCK")
registerDoSNOW(cl)

# Methods for learning
set.seed(72342)
# Kth nearest neigh
fit.knn <- train(Ratio~., data=input_data, method="knn", trControl=control)
# r part 
fit.cart <- train(Ratio~., data=input_data, method="rpart", trControl=control)
# vector machines
fit.svm <- train(Ratio~., data=input_data, method="svmRadial", trControl=control)
# random forrest
fit.rf <- train(Ratio~., data=input_data, method="rf", trControl=control)

stopCluster(cl)

total.time <- start.time - Sys.time()
total.time
# 
results <- resamples(list(rf = fit.rf, knn=fit.knn, svm = fit.svm, cart = fit.cart))
summary(results)

dotplot(results)

print(fit.rf)



