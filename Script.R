getwd()
setwd("D:/Rbag")
install.packages('MLmetrics',destdir="D:/Rbag")
install.packages('ggplot2',destdir="D:/Rbag")
install.packages('A3',destdir="D:/Rbag")
install.packages('reshape2',destdir="D:/Rbag")
install.packages('glmnet',destdir="D:/Rbag")
install.packages('caret',destdir="D:/Rbag")
install.packages('e1071',destdir="D:/Rbag")
install.packages('tidyverse',destdir="D:/Rbag")
library(tidyverse)
library(e1071)
library(broom)
library(caret)
library(kknn)
library(reshape2)
library(naiveBayes)
library(randomForest)
library(glmnet)
library(nnet)
library(xgboost)

#Machine learning calssification models optimization 
otu1 <- read.csv('presence data of source.csv', row.names = 1)
#otu1 <- read.csv('relative abundance data of source.csv', row.names = 1)
#otu1 <- read.csv('count data of source.csv', row.names = 1)
#otu1 <- read.csv('log-relative abundance data of source.csv', row.names = 1)
#otu1 <- read.csv('log-count data of source', row.names = 1)


otu2 <- read.csv('presence data of stimulated sink.csv', row.names = 1)
#otu2 <- read.csv('relative abundance data of stimulated sink.csv', row.names = 1)
#otu2 <- read.csv('count data of stimulated sink.csv', row.names = 1)
#otu2 <- read.csv('log-relative abundance data of stimulated sink.csv', row.names = 1)
#otu2 <- read.csv('log-count data of stimulated sink.csv', row.names = 1)


otu1$Group<- as.factor(otu1$Group)
otu2$Group<- as.factor(otu2$Group)

set.seed(111)
trainlist <- caret::createDataPartition(otu1$Group,p = 0.7, list = FALSE)
otu.train <- otu1[trainlist,]
otu.test <- otu1[-trainlist,]
otu.valid <- otu2



set.seed(111)
k_values <- seq(1, 30, by = 1)  # Example: testing k from 1 to 20  
results <- data.frame(k = integer(), accuracy = numeric(), stringsAsFactors = FALSE)  

# Perform kNN with cross-validation to determine the best k  
for (k in k_values) {  
  # Train kNN model with the current value of k  
  knn_train <- kknn(Group ~ ., train = otu.train, test = otu.test, k = k, distance = 2)  # You may also try different distances  
  
  # Get the fitted values  
  knn_test <- knn_train$fitted.values  
  
  # Calculate accuracy  
  accuracy <- sum(knn_test == otu.test$Group) / length(otu.test$Group)  
  
  # Store the results  
  results <- rbind(results, data.frame(k = k, accuracy = accuracy))  
}  

best_results <- results[which.max(results$accuracy),]  
cat("Best k:", best_results$k, "with Accuracy:", best_results$accuracy, "\n")  

train_knn <- kknn(Group ~ ., train = otu.train, test = otu.train, k = best_results$k, distance = 2)  
train_knn_test <- train_knn$fitted.values  

# Confusion matrix for final predictions  
conf_matrix <- caret::confusionMatrix(as.factor(train_knn_test), otu.train$Group)  
print(conf_matrix) 

# Train the final KNN model using the best k  
final_knn <- kknn(Group ~ ., train = otu.train, test = otu.test, k = best_results$k, distance = 2)  
final_knn_test <- final_knn$fitted.values  

# Confusion matrix for final predictions  
conf_matrix <- caret::confusionMatrix(as.factor(final_knn_test), otu.test$Group)  
print(conf_matrix)  

# Validate on the validation set  
final_knn_valid <- kknn(Group ~ ., train = otu.train, test = otu.valid, k = best_results$k, distance = 2)  
final_knn_valid <- final_knn_valid$fitted.values  

# Confusion matrix for validation predictions  
conf_matrix_valid <- caret::confusionMatrix(as.factor(final_knn_valid), otu.valid$Group)  
print(conf_matrix_valid)  

#LR
set.seed(111)  

model <- glm(Group ~ ., data = otu.train, family = binomial(link = "logit"))  
summary(model)
prob_logit <- predict(model, newdata = otu.test, type = "response")  



# Define a range of thresholds to test  
thresholds <- seq(0, 2, by = 0.01)  
accuracy_results <- data.frame(threshold = thresholds, accuracy = NA)  

# Iterate over thresholds to calculate accuracy  
for (i in seq_along(thresholds)) {  
  pre_logit <- ifelse(prob_logit > thresholds[i], levels(otu.train$Group)[2], levels(otu.train$Group)[1])  
  accuracy_results$accuracy[i] <- mean(otu.test$Group == pre_logit)  
}  

# Find the best threshold  
best_threshold <- accuracy_results$threshold[which.max(accuracy_results$accuracy)]  
best_accuracy <- max(accuracy_results$accuracy)  

# Print the best threshold and accuracy  
cat("Best Threshold:", best_threshold, "\n")  
cat("Best Accuracy:", best_accuracy, "\n")  

pre_logit_final <- ifelse(prob_logit > best_threshold, levels(otu.train$Group)[2], levels(otu.train$Group)[1])  

# Create the confusion matrix for test predictions  
group_cf <- caret::confusionMatrix(as.factor(pre_logit_final), otu.test$Group)  
print(group_cf)

str(otu.train)  
summary(otu.train)  
print(table(otu.train$Group))  

# Fit the full model  
model <- tryCatch({  
  glm(Group ~ ., data = otu.train, family = binomial(link = "logit"))  
}, error = function(e) {  
  cat("Error in model fitting:", e$message, "\n")  
  return(NULL)  
})  


# If the model was successfully fitted, inspect coefficients  
if (!is.null(model)) {  
  summary(model)  
  coef_values <- coef(model)  
  print(coef_values)  
}

coef_values <- coef(model)  

# Check the coefficients to diagnose any issues  
print(coef_values)  
str(coef_values)  # This shows the structure of coefficients  


# Validation predictions  
prob_logit_valid <- predict(model, newdata = otu.valid, type = "response")  
pre_logit_final_valid <- ifelse(prob_logit_valid > best_threshold, levels(otu.train$Group)[2], levels(otu.train$Group)[1])  
group_cf_valid <- caret::confusionMatrix(as.factor(pre_logit_final_valid), otu.valid$Group)  
print(group_cf_valid)  

#xgboost
train_y <- as.numeric(as.factor(otu.train[, 1])) - 1  # Convert to numeric and subtract 1 to get 0/1  
train_data1 <- otu.train[, c(2:6)]  
train <- xgb.DMatrix(data = as.matrix(train_data1), label = train_y)  

test_data1 <- otu.test[, c(2:6)]  
test_y <- as.numeric(as.factor(otu.test[, 1])) - 1  # Same conversion for test data  
test <- xgb.DMatrix(data = as.matrix(test_data1), label = test_y)  

valid_data1 <- otu.valid[, c(2:6)]  
valid_y <- as.numeric(as.factor(otu.valid[, 1])) - 1  # Same conversion for test data  
valid <- xgb.DMatrix(data = as.matrix(valid_data1), label = valid_y) 


# Define a grid of hyperparameters  
param_grid <- expand.grid(
  max_depth = c(1:10),
  eta = c(0.01:1),
  nrounds = c(50:200),
  gamma = c(0:1),  
  subsample = c(0:1),  
  colsample_bytree = c(0:1)  
) 
# Initialize a data frame to store results  
results <- data.frame(max_depth = integer(),
                      eta = numeric(),
                      nrounds = integer(),
                      gamma = numeric(),
                      subsample = numeric(),
                      colsample_bytree = numeric(),
                      accuracy = numeric())  

# Loop over all combinations of hyperparameters  
  for (i in 1:nrow(param_grid)) {
    params <- list(
      objective = "binary:logistic",  
      eval_metric = "logloss",
      eta = param_grid$eta[i],
      max_depth = param_grid$max_depth[i],
      gamma = param_grid$gamma[i],
      subsample = param_grid$subsample[i],
      colsample_bytree = param_grid$colsample_bytree[i]
    )  
  # Train the model with current parameters  
  model <- xgboost(params = params, data = train, nrounds = param_grid$nrounds[i], verbose = 0)  
   # Predict on the training set  
  train_predict <- predict(model, newdata = train)  
  train_prediction <- ifelse(train_predict > 0.5, 1, 0)  # Predict as 1 if probability > 0.5  
  
  # Calculate accuracy  
  accuracy <- mean(train_prediction == train_y)  
  
  # Store the results  
  results <- rbind(results, data.frame(max_depth = param_grid$max_depth[i],
                                       eta = param_grid$eta[i],
                                       nrounds = param_grid$nrounds[i],
                                       gamma = param_grid$gamma[i],
                                       subsample = param_grid$subsample[i],
                                       colsample_bytree = param_grid$colsample_bytree[i],
                                       accuracy = accuracy))
  }  


# Find the best hyperparameters based on accuracy  
best_params <- results[which.max(results$accuracy),]  
cat("Best Parameters:\n")  
print(best_params)  

# Train the final model with the best parameters  
final_model <- xgboost(params = list(
  objective = "binary:logistic",  
  eval_metric = "logloss",
  eta = best_params$eta,
  max_depth = best_params$max_depth,
  gamma = best_params$gamma,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree
), data = train, nrounds = best_params$nrounds)  

# Make final predictions on the test data
train_predict<-predict(final_model, newdata = train) 
write.table(train_predict, '2.txt', sep = '\t', col.names = NA, quote = FALSE)
train_prediction <- ifelse(train_predict > 0.5, 1, 0)
train_accuracy <- mean(train_prediction == train_y)  
print(paste("Test Accuracy:", train_accuracy))  

# Confusion matrix  
group_cf <- caret::confusionMatrix(as.factor(train_prediction), as.factor(train_y))  
print(group_cf)

test_predict <- predict(final_model, newdata = test)  
test_prediction <- ifelse(test_predict > 0.5, 1, 0)  # Predict as 1 if probability > 0.5  

# Calculate test accuracy  
test_accuracy <- mean(test_prediction == test_y)  
print(paste("Test Accuracy:", test_accuracy))  

# Confusion matrix  
group_cf <- caret::confusionMatrix(as.factor(test_prediction), as.factor(test_y))  
print(group_cf)

valid_predict <- predict(final_model, newdata = valid)  
valid_prediction <- ifelse(valid_predict > 0.5, 1, 0)  # Predict as 1 if probability > 0.5  
valid_prediction
write.table(valid_predict, '1.txt', sep = '\t', col.names = NA, quote = FALSE)
# Calculate test accuracy  
valid_accuracy <- mean(valid_prediction == valid_y)  
print(paste("Valid Accuracy:", valid_accuracy))  

# Confusion matrix  
group_cf_valid <- caret::confusionMatrix(as.factor(valid_prediction), as.factor(valid_y))  
print(group_cf_valid)

#NB
laplace_values <- seq(0, 10, by = 0.01) # Possible values for Laplace smoothing  
accuracy_results <- data.frame(laplace = laplace_values, accuracy = NA)  

# Train Naive Bayes model for each Laplace value and calculate accuracy  
for (i in seq_along(laplace_values)) {  
  # Train model  
  naive <- naiveBayes(Group ~ ., data = otu.train, laplace = laplace_values[i])  
  
  # Make predictions  
  predict <- predict(naive, newdata = otu.test, type = "class")  
  
  # Calculate accuracy  
  accuracy_results$accuracy[i] <- mean(otu.test$Group == predict)  
}  

# Find the best Laplace value corresponding to the maximum accuracy  
best_laplace <- accuracy_results$laplace[which.max(accuracy_results$accuracy)]  
best_accuracy <- max(accuracy_results$accuracy)  

# Print best Laplace value and corresponding accuracy  
cat("Best Laplace:", best_laplace, "\n")  
cat("Best Accuracy:", best_accuracy, "\n")  

# Train final model with the best Laplace value  
final_naive <- naiveBayes(Group ~ ., data = otu.train, laplace = best_laplace)  
final_prob_predict <- predict(final_naive, newdata = otu.test, type = "raw")  

# Final predictions using the best model  
final_predict <- predict(final_naive, newdata = otu.test, type = "class")  

# Create the confusion matrix  
group_cf <- caret::confusionMatrix(as.factor(final_predict), otu.test$Group)  
print(group_cf)

final_prob_predict_valid <- predict(final_naive, newdata = otu.valid, type = "raw") 
final_predict_valid <- predict(final_naive, newdata = otu.valid, type = "class")  

# Create the confusion matrix  
group_cf_valid <- caret::confusionMatrix(as.factor(final_predict_valid), otu.valid$Group)  
print(group_cf_valid)

#RF
tuneGrid <- expand.grid(mtry = 1:10) # Range for mtry  

# Train a random forest model with hyperparameter tuning on mtry  
rf_tune <- train(Group ~ ., data = otu.train, method = "rf",  
                 tuneGrid = tuneGrid,  
                 trControl = trainControl(method = "cv", number = 10))  # 5-fold cross-validation  

# Display the best mtry value and corresponding results  
best_mtry <- rf_tune$bestTune$mtry  
print(best_mtry)  
print(rf_tune)  

# Now train the final random forest model with the best mtry and specify ntree separately  
ntree_values <- c(100, 300, 500, 1000, 3000)  # Set of ntree values to test  
model_results <- list()  # To store results for different ntree values  

# Loop over the different values of ntree and calculate the performance  
for (ntree in ntree_values) {  
  rf_model <- randomForest(Group ~ ., data = otu.train,   
                           mtry = best_mtry,   
                           ntree = ntree,   
                           importance = TRUE,   
                           proximity = TRUE)  
  
  # Store the model and performance metrics  
  model_results[[as.character(ntree)]] <- list(  
    model = rf_model,  
    confusionMatrix = caret::confusionMatrix(predict(rf_model, otu.train), otu.train$Group)  
  )  
  
  # Print confusion matrix for training set  
  print(paste("NTREE:", ntree))  
  print(model_results[[as.character(ntree)]]$confusionMatrix)  
}  

# Select the best model based on training performance and rebuild for final predictions  
best_ntree_model <- model_results[[as.character(ntree_values[which.max(sapply(model_results, function(x) x$confusionMatrix$overall['Accuracy']))])]]$model  

# Predictions on the test set using the best model configured  
group_predict_prob <- predict(best_ntree_model, otu.test, type = "prob") 
group_predict <- predict(best_ntree_model, otu.test)  

# Plot predictions on the test set  
plot(otu.test$Group, group_predict, main = '测试集 (Test Set)',  
     xlab = 'Group', ylab = 'Predicted Group')  
abline(0, 1)  # Ideal line for perfect prediction  

# Confusion Matrix for test set predictions  
group_cf_test <- caret::confusionMatrix(group_predict, otu.test$Group)  
print(group_cf_test) 

group_predict_valid_prob <- predict(best_ntree_model, otu.valid, type = "prob") 
group_predict_valid <- predict(best_ntree_model, otu.valid)  

# Plot predictions on the test set  
plot(otu.valid$Group, group_predict_valid, main = '测试集 (Test Set)',  
     xlab = 'Group', ylab = 'Predicted Group')  
abline(0, 1)  # Ideal line for perfect prediction  

# Confusion Matrix for test set predictions  
group_cf_test_valid <- caret::confusionMatrix(group_predict_valid, otu.valid$Group)  
print(group_cf_test_valid) 

#ANN
set.seed(111)

# Define control for train function  
train_control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation  

# Set up a grid of hyperparameters to tune  
tune_grid <- expand.grid(
  size = c(1:30), 
  decay = c(0.001:1)
)

# Train the neural network model with tuning  
nn_model <- train(Group ~., data = otu.train,   
                  method = "nnet",   
                  trControl = train_control,   
                  tuneGrid = tune_grid,   
                  linout = FALSE,  
                  trace = FALSE) # Set trace to FALSE to suppress training output  

# View the best model parameters  
cat("Best Model Parameters:\n")  
print(nn_model$bestTune)  
plot(nn_model)
print(nn_model)  

final_model <- nnet(Group ~., data = otu.train, 
                    size = nn_model$bestTune$size,   
                    decay = nn_model$bestTune$decay, 
                    linout = FALSE, 
                    trace = FALSE,
                    maxit = 500, 
                    abstol = 0.001 
)  


predictions <- predict(final_model, newdata = otu.train, type = "class") 
predictions

accuracy <- sum(predictions == otu.test$Group) / length(predictions)
cat("Accuracy:", accuracy, "\n")

confusion_matrix <- caret::confusionMatrix(as.factor(predictions), otu.test$Group)
print(confusion_matrix)

predictions <- predict(final_model, newdata = otu.test, type = "class") 
predictions

accuracy <- sum(predictions == otu.valid$Group) / length(predictions)
cat("Accuracy:", accuracy, "\n")

confusion_matrix <- caret::confusionMatrix(as.factor(predictions), otu.valid$Group)
print(confusion_matrix)


#linear
set.seed(111)
train_control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation
#tune_grid <- expand.grid(C = c(0.1:100)) # Cost parameter for linear SVM  
tune_grid <- expand.grid(C = seq(0.1, 10, by = 0.01))
svm_model <- train(Group ~ ., data = otu.train,   
                   method = "svmLinear",  
                   trControl = train_control,   
                   tuneGrid = tune_grid,  
                   probability = TRUE)  
svm_model
# Make predictions on the test set 
#train<-svm(Group ~ ., data = otu.train,kernel="linear",C=0.97, probability = TRUE)
p_test <- predict(svm_model, otu.test) 
#p_test_prob <- attr(predict(train,otu.test, probability = TRUE), "probabilities") 
t<-table(p_test,otu.test$Group)
t
acc<-sum(diag(t))/nrow(otu.test)
acc
confusion_matrix <- caret::confusionMatrix(as.factor(p_test), otu.test$Group)
print(confusion_matrix)

# Make predictions on the validation set
p_valid <- predict(svm_model, otu.valid)
p_valid
t<-table(p_valid,otu.valid$Group)
t
acc<-sum(diag(t))/nrow(otu.valid)
acc
# Confusion matrix for validation set  
confusion_matrix_valid <- caret::confusionMatrix(as.factor(p_valid), otu.valid$Group)
print(confusion_matrix_valid)

#radial
set.seed(666)
train_control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation
tune_grid <- expand.grid(C = c(0.1: 100),  # Cost parameter  
                         sigma = c(0.01:1)) # Sigma for RBF kernel  
svm_model <- train(Group ~ ., data = otu.train,   
                   method = "svmRadial",  
                   trControl = train_control,   
                   tuneGrid = tune_grid,  
                   probability = TRUE)  
svm_model

p_test <- predict(svm_model, otu.test) 

t<-table(p_test,otu.test$Group)
t
acc<-sum(diag(t))/nrow(otu.test)
acc
confusion_matrix <- caret::confusionMatrix(as.factor(p_test), otu.test$Group)
print(confusion_matrix)

p_valid <- predict(svm_model, otu.valid)
p_valid
t<-table(p_valid,otu.valid$Group)
t
acc<-sum(diag(t))/nrow(otu.valid)
acc
# Confusion matrix for validation set  
confusion_matrix_valid <- caret::confusionMatrix(as.factor(p_valid), otu.valid$Group)
print(confusion_matrix_valid)

#sigmoid
set.seed(111)
train_control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation

# Train the SVM model using the sigmoid kernel  
tune_results <- tune(svm, Group ~ ., data = otu.train,  
                     kernel = "sigmoid", # Specify sigmoid kernel  
                     ranges = list(cost = seq(0.1, 10, by = 0.5),  # Cost parameter  
                                   gamma = seq(0.01, 10, by = 0.1))) # Gamma parameter  

# View the best hyperparameter configuration
best_params <- tune_results$best.parameters

svm_model <- svm(Group ~., data = otu.train,  
                 kernel = "sigmoid",  
                 cost = best_params$cost,  
                 gamma = best_params$gamma,  
                 probability = TRUE)


# Make predictions on the test set  
p <- predict(svm_model, newdata = otu.test)
p
t<-table(p,otu.test$Group)
t
acc<-sum(diag(t))/nrow(otu.test)
acc
confusion_matrix <- caret::confusionMatrix(as.factor(p), otu.test$Group)
print(confusion_matrix)

p_valid <- predict(svm_model, otu.valid)
p_valid
t<-table(p_valid,otu.valid$Group)
t
acc<-sum(diag(t))/nrow(otu.valid)
acc
# Confusion matrix for validation set  
confusion_matrix_valid <- caret::confusionMatrix(as.factor(p_valid), otu.valid$Group)
print(confusion_matrix_valid)

#polynomial
set.seed(111)
train_control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation
tune_grid <- expand.grid(degree = c(1,2,3,4,5,6),  # Degree of the polynomial  
                         scale = c(0.01,0.1,1,10), # Scaling factor (for radial basis function, but can try for polynomial)  
                         C = c(0.1, 1,10,100)) # Cost parameter 

svm_model <- train(Group ~ ., data = otu.train,   
                   method = "svmPoly",  
                   trControl = train_control,   
                   tuneGrid = tune_grid,
                   probability = TRUE)  
svm_model
# Make predictions on the test set 
p_test <- predict(svm_model, otu.test, probability = TRUE) 
t<-table(p_test,otu.test$Group)
t
acc<-sum(diag(t))/nrow(otu.test)
acc
confusion_matrix <- caret::confusionMatrix(as.factor(p_test), otu.test$Group)
print(confusion_matrix)

# Make predictions on the validation set
p_valid <- predict(svm_model, otu.valid)
p_valid
t<-table(p_valid,otu.valid$Group)
t
acc<-sum(diag(t))/nrow(otu.valid)
acc
# Confusion matrix for validation set  
confusion_matrix_valid <- caret::confusionMatrix(as.factor(p_valid), otu.valid$Group)
print(confusion_matrix_valid)


#Model ensemble and identification thresholds
otu3 <- read.csv('presence data of source.csv', row.names = 1)
#otu3 <- read.csv('relative abundance data of source.csv', row.names = 1)
otu4 <- read.csv('presence data of stimulated sink.csv', row.names = 1)
#otu4 <- read.csv('relative abundance data of stimulated sink.csv', row.names = 1)


#otu4 <- read.csv('presence data of identification thresholds.csv', row.names = 1)
#otu4 <- read.csv('f__AKAU3564_sediment_group relative abundance identification thresholds.csv', row.names = 1)
#otu4 <- read.csv('f__Desulfuromonadaceae relative abundance identification thresholds.csv', row.names = 1)
#otu4 <- read.csv('g__Geobacter relative abundance identification thresholds.csv', row.names = 1)
#otu4 <- read.csv('o__Dehalococcoidales relative abundance identification thresholds.csv', row.names = 1)
#otu4 <- read.csv('f__Desulfuromonadaceae-g__Geobacter relative abundance identification thresholds.csv', row.names = 1)
otu3$Group<- as.factor(otu3$Group)
otu4$Group<- as.factor(otu4$Group)


otu.train <- otu3
otu.test <- otu4

#XGBoost
train_y <- as.numeric(as.factor(otu.train[, 1])) - 1  # Convert to numeric and subtract 1 to get 0/1  
train_data1 <- otu.train[, c(2:6)]  
train <- xgb.DMatrix(data = as.matrix(train_data1), label = train_y)  

test_data1 <- otu.test[, c(2:6)]  
test_y <- as.numeric(as.factor(otu.test[, 1])) - 1  # Same conversion for test data  
test <- xgb.DMatrix(data = as.matrix(test_data1), label = test_y)  

# Define a grid of hyperparameters  
param_grid <- expand.grid(
  max_depth = c(6),
  eta = c(0.01),
  nrounds = c(172),
  gamma = c(0),  
  subsample = c(1),  
  colsample_bytree = c(1)  
) 
# Initialize a data frame to store results  
results <- data.frame(max_depth = integer(),
                      eta = numeric(),
                      nrounds = integer(),
                      gamma = numeric(),
                      subsample = numeric(),
                      colsample_bytree = numeric(),
                      accuracy = numeric())  

# Loop over all combinations of hyperparameters  
for (i in 1:nrow(param_grid)) {
  params <- list(
    objective = "binary:logistic",  
    eval_metric = "logloss",
    eta = param_grid$eta[i],
    max_depth = param_grid$max_depth[i],
    gamma = param_grid$gamma[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i]
  )  
  # Train the model with current parameters  
  model <- xgboost(params = params, data = train, nrounds = param_grid$nrounds[i], verbose = 0)  
  # Predict on the training set  
  train_predict <- predict(model, newdata = train)  
  train_prediction <- ifelse(train_predict > 0.5, 1, 0)  # Predict as 1 if probability > 0.5  
  
  # Calculate accuracy  
  accuracy <- mean(train_prediction == train_y)  
  
  # Store the results  
  results <- rbind(results, data.frame(max_depth = param_grid$max_depth[i],
                                       eta = param_grid$eta[i],
                                       nrounds = param_grid$nrounds[i],
                                       gamma = param_grid$gamma[i],
                                       subsample = param_grid$subsample[i],
                                       colsample_bytree = param_grid$colsample_bytree[i],
                                       accuracy = accuracy))
}  


# Find the best hyperparameters based on accuracy  
best_params <- results[which.max(results$accuracy),]  
cat("Best Parameters:\n")  
print(best_params)  

# Train the final model with the best parameters  
final_model <- xgboost(params = list(
  objective = "binary:logistic",  
  eval_metric = "logloss",
  eta = best_params$eta,
  max_depth = best_params$max_depth,
  gamma = best_params$gamma,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree
), data = train, nrounds = best_params$nrounds)  

# Make final predictions on the test data
train_predict<-predict(final_model, newdata = train) 
write.table(train_predict, '2.txt', sep = '\t', col.names = NA, quote = FALSE)
train_prediction <- ifelse(train_predict > 0.5, 1, 0)
train_accuracy <- mean(train_prediction == train_y)  
print(paste("Test Accuracy:", train_accuracy))  

# Confusion matrix  
group_cf <- caret::confusionMatrix(as.factor(train_prediction), as.factor(train_y))  
print(group_cf)

test_predict <- predict(final_model, newdata = test)  
test_prediction <- ifelse(test_predict > 0.5, 1, 0)  # Predict as 1 if probability > 0.5  

# Calculate test accuracy  
test_accuracy <- mean(test_prediction == test_y)  
print(paste("Test Accuracy:", test_accuracy))  

# Confusion matrix  
group_cf <- caret::confusionMatrix(as.factor(test_prediction), as.factor(test_y))  
print(group_cf)


importance_matrix <- xgb.importance(feature_names = colnames(train_data1), model = final_model)  

# Print the importance matrix  
print(importance_matrix)  

# Plot absolute feature importance using ggplot2   
# Here we take the raw importance values directly  
importance_df <- as.data.frame(importance_matrix)  
importance_df <- importance_df[order(-importance_df$Gain), ]  # Order by importance  
importance_df
# Create a bar plot of feature importance  
ggplot(importance_df, aes(x = reorder(Feature, Gain), y = Gain)) +  
  geom_bar(stat = "identity", fill = "steelblue") +  
  coord_flip() +  
  theme_minimal() +  
  labs(title = "Feature Importance for XGBoost Model",  
       x = "Features",  
       y = "Importance (Gain)") 

#ANN
set.seed(111)

# Define control for train function  
train_control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation  

# Set up a grid of hyperparameters to tune  
tune_grid <- expand.grid(
  size = c(1), 
  decay = c(0.001)
)

# Train the neural network model with tuning  
nn_model <- train(Group ~., data = otu.train,   
                  method = "nnet",   
                  trControl = train_control,   
                  tuneGrid = tune_grid,   
                  linout = FALSE,  
                  trace = FALSE) # Set trace to FALSE to suppress training output  

# View the best model parameters  
cat("Best Model Parameters:\n")  
print(nn_model$bestTune)  
plot(nn_model)
print(nn_model)  

final_model <- nnet(Group ~., data = otu.train, 
                    size = nn_model$bestTune$size,   
                    decay = nn_model$bestTune$decay, 
                    linout = FALSE, 
                    trace = FALSE,
                    maxit = 500, 
                    abstol = 0.001 
)  


# Make predictions on the validation set  
#predictions_prob <- predict(final_model, newdata = otu.test, type = "raw") 

# Check the structure of predictions_prob  
calculate_permutation_importance <- function(model, data, target_col, n_perm = 100) {  
  actual_accuracy <- sum(predict(model, newdata = data, type = "class") == data[[target_col]]) / nrow(data)  
  importance <- sapply(names(data)[-which(names(data) == target_col)], function(feature) {  
    permuted_accuracy <- replicate(n_perm, {  
      permuted_data <- data  
      permuted_data[[feature]] <- sample(permuted_data[[feature]])  
      sum(predict(model, newdata = permuted_data, type = "class") == permuted_data[[target_col]]) / nrow(permuted_data)  
    })  
    actual_accuracy - mean(permuted_accuracy)  
  })  
  importance  
}  

# Calculate feature importance  
feature_importance <- calculate_permutation_importance(final_model, otu.train, "Group")  

# Create a data frame for visualization  
importance_df <- data.frame(Feature = names(feature_importance),   
                            Importance = feature_importance)  

# Sorting importance  
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ] 
importance_df

predictions <- predict(final_model, newdata = otu.train, type = "class") 
predictions

accuracy <- sum(predictions == otu.test$Group) / length(predictions)
cat("Accuracy:", accuracy, "\n")

confusion_matrix <- caret::confusionMatrix(as.factor(predictions), otu.test$Group)
print(confusion_matrix)

predictions <- predict(final_model, newdata = otu.test, type = "class") 
predictions

accuracy <- sum(predictions == otu.test$Group) / length(predictions)
cat("Accuracy:", accuracy, "\n")

confusion_matrix <- caret::confusionMatrix(as.factor(predictions), otu.test$Group)
print(confusion_matrix)

