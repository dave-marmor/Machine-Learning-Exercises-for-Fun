library(dplyr)
library(GGally)
library(ggplot2)
library(caret)

#### Import the data ####
df <- iris

#### Explore the data #####
str(df)
summary(df)
ggpairs(df)

# Test all possible combos of 2 dimensional clusters
ggplot(data = df, aes(x = Sepal.Length, y = Sepal.Width)) + geom_point(aes(color = Species))
ggplot(data = df, aes(x = Sepal.Length, y = Petal.Length)) + geom_point(aes(color = Species))
ggplot(data = df, aes(x = Sepal.Length, y = Petal.Width)) + geom_point(aes(color = Species))
ggplot(data = df, aes(x = Sepal.Width, y = Petal.Length)) + geom_point(aes(color = Species))
ggplot(data = df, aes(x = Sepal.Width, y = Petal.Width)) + geom_point(aes(color = Species))
ggplot(data = df, aes(x = Petal.Length, y = Petal.Width)) + geom_point(aes(color = Species))


#### Prepare the data for modeling ####
set.seed(1111)

# Partition the data into train and test sets
split <- sample(1:nrow(df), .75 * nrow(df))
train_set <- df[split, ]
test_set <- df[-split, ]
X_train = train_set %>%
  select(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)
y_train <- train_set$Species
X_test = test_set %>%
  select(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)
y_test <- test_set$Species

# Scale the values
scale_values <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(scale_values, X_train)
X_test_scaled <- predict(scale_values, X_test)


#### Model 1 - KNN ####
library(class)

# Train the model
set.seed(2222)
knn_pred <- knn(train = X_train_scaled, test = X_test_scaled,
              cl = y_train,  k = 3) 
knn_pred_summary <- test_set %>%
  mutate(y_pred = knn_pred)
head(knn_pred_summary)

# Confusion Matrix
knn_conf_mat <- table(Actual = y_test, Predicted = knn_pred)
knn_conf_mat

# Accuracy
knn_accuracy <- sum(diag(knn_conf_mat)) / sum(knn_conf_mat)
knn_accuracy

#### Model 2 - SVM ####
library(e1071)

# Train the model
svm_model <- svm(x = X_train_scaled, y = y_train, scale = FALSE,
                 type = "C-classification", kernel = "radial")

# Predict using the test set
svm_pred <- predict(svm_model, newdata = X_test_scaled)
svm_pred_summary <- test_set %>%
  mutate(y_pred = svm_pred)
head(svm_pred_summary)

# Confusion Matrix
svm_conf_mat <- table(Actual = y_test, Predicted = svm_pred)
svm_conf_mat

# Accuracy
svm_accuracy <- sum(diag(svm_conf_mat)) / sum(svm_conf_mat)
svm_accuracy

#### Model 3 - xgboost ####
library(xgboost)

# Convert dataframes to matrices
X_train_mat <- data.matrix(X_train)
y_train_num <- as.numeric(y_train) - 1
X_test_mat <- data.matrix(X_test)
y_test_num <- as.numeric(y_test) - 1
labels <- levels(y_train)

# Train the model
set.seed(2222)
xgb_model <- xgboost(data = X_train_mat, label = y_train_num, nrounds = 10,
                     objective = "multi:softprob", num_class = 3)

# Predict using test set
y_prob = predict(xgb_model, newdata = X_test_mat, reshape = TRUE)
y_pred = labels[max.col(y_prob)]
xgb_pred_summary <- cbind(test_set, y_prob, y_pred)
names(xgb_pred_summary) = c(names(test_set), paste0("y_prob_", labels), "y_pred")
head(xgb_pred_summary)

# Confusion Matrix & Accuracy
xgb_conf_mat <- table(Actual = y_test, Predicted = y_pred)
xgb_conf_mat
xgb_accuracy <- sum(diag(xgb_conf_mat)) / sum(xgb_conf_mat)
xgb_accuracy


#### Summary of Model Accuracies ####
c(knn = round(knn_accuracy, 2), svm = round(svm_accuracy, 2), xgb = round(xgb_accuracy, 2))
