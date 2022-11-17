###############################################################################################
# ANALYSIS AND MACHINE LEARNING PREDICTIONS FROM BIOMECHANICAL FEATURES OF ORTHOPEDIC PATIENTS
###############################################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(rpart)
library(randomForest)


##############################################################
# Load the Biomechanical Features of Orthopedic Patients data
##############################################################

# Biomechanical Features of Orthopedic Patients dataset:
# https://github.com/mariacaldeira/data-science-hw

url <- "https://raw.githubusercontent.com/mariacaldeira/data-science-hw/master/column_2C_weka.csv"
column_2C_weka <- read.csv(url)

url2 <- "https://raw.githubusercontent.com/mariacaldeira/data-science-hw/master/column_3C_weka.csv"
column_3C_weka <- read.csv(url2)


####################################################################################################
# 1st. Task - Classify patients as belonging to one out of three categories: 
#             Normal (100 patients), Disk Hernia (60 patients) or Spondylolisthesis (150 patients)
####################################################################################################


####### Plot of the Counting from each Class
column_3C_weka %>%
  group_by(class) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = class, y = count)) +
  geom_point()


####### Validation set will be 20% of the dataset
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = column_3C_weka$class, times = 1, p = 0.2, list = FALSE)
train_set <- column_3C_weka[-test_index,]
test_set <- column_3C_weka[test_index,]

#Verification of how many rows in each dataset and how many from each class in order to assure we have enough rows from each class
dim(train_set)
dim(test_set)

train_set %>%
  group_by(class) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = class, y = count)) +
  geom_point()

test_set %>%
  group_by(class) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = class, y = count)) +
  geom_point()

train_set %>%
  group_by(class) %>%
  summarize(count = n())

test_set %>%
  group_by(class) %>%
  summarize(count = n())

########### Trying Random Guessing

#set.seed(1991)
set.seed(1991, sample.kind = "Rounding")
# guess with equal probability the class
guess <- sample(c("Hernia", "Normal","Spondylolisth"), nrow(test_set), replace = TRUE)
mean(guess == test_set$class)


########### Fit a Classification Tree Model

x_train <- subset(train_set, select = -c(class))
y_train <- train_set[,7]
x_test <- subset(test_set, select = -c(class))
y_test <- test_set[,7]

# Method 1: Classification Tree Model with the best cp value and standard minsplit (20 observations before splitting)

# set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later

#Training Model
fit_rpart <- with(column_3C_weka, 
            train(x_train, y_train, method = "rpart",
                  tuneGrid = data.frame(cp = seq(0, 0.10, 0.01))))

ggplot(fit_rpart)
confusionMatrix(fit_rpart)

#Predicting test_set
p_hat_rpart <- predict(fit_rpart, test_set)
y_hat_rpart <- factor(p_hat_rpart)

accuracy1 <- data.frame("res" = confusionMatrix(y_hat_rpart, factor(test_set$class))$overall[["Accuracy"]])
accuracy1

# make plot of decision tree
plot(fit_rpart$finalModel, margin = 0.1)
text(fit_rpart$finalModel)

# Method 2: Classification Tree Model with the best cp value and minsplit = 0
# set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later

#Training Model
fit_rpart2 <- with(column_3C_weka, 
                  train(x_train, y_train, method = "rpart",
                        tuneGrid = data.frame(cp = seq(0, 0.10, 0.01)),
                        control = rpart.control(minsplit = 0)))

ggplot(fit_rpart2)
confusionMatrix(fit_rpart2)

#Predicting test_set
p_hat_rpart2 <- predict(fit_rpart2, test_set)
y_hat_rpart2 <- factor(p_hat_rpart2)

accuracy2 <- data.frame("res" = confusionMatrix(y_hat_rpart2, factor(test_set$class))$overall[["Accuracy"]])
accuracy2


### Comparing those Methods the chosen one was the Method 1 - With that the best tune from this method is = 0.02 & Final Accuracy from Classification Tree Model using the test dataset
fit_rpart$bestTune
#Final Accuracy
accuracy1


########### Fit a Random Forest Model

# set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later

#Training Model
fit_rf <- with(column_3C_weka, 
          train(x_train, y_train, method = "rf", 
          nodesize = 1,
          tuneGrid = data.frame(mtry = seq(50, 200, 25))))

ggplot(fit_rf)
confusionMatrix(fit_rf)

#Predicting test_set
p_hat_rf <- predict(fit_rf, test_set)
y_hat_rf <- factor(p_hat_rf)

accuracy_rf <- data.frame("res" = confusionMatrix(y_hat_rf, factor(test_set$class))$overall[["Accuracy"]])
accuracy_rf

#Variables of Importance
imp <- varImp(fit_rf)
imp

#Calculating the Variables of Importance from the Random Forest Model

tree_terms <- as.character(unique(fit_rpart$finalModel$frame$var[!(fit_rpart$finalModel$frame$var == "<leaf>")]))
tree_terms

data_frame(term = rownames(imp$importance), 
           importance = imp$importance$Overall) %>%
  mutate(rank = rank(-importance)) %>% arrange(desc(importance)) %>%
  filter(term %in% tree_terms)

#Verification if those two main variables are correlated to each other:
cor(column_3C_weka$degree_spondylolisthesis, column_3C_weka$sacral_slope)

########### Fit a LDA Model

#set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later
fit_lda <- train(x_train, y_train, method = "lda", data = train_set)
y_hat_lda <- predict(fit_lda, test_set)
accuracy_lda <- mean(y_hat_lda == test_set$class)



########## Fit KNN Model

#set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later
fit_knn <- train(class ~ .,
                   method = "knn",
                   data = train_set,
                   tuneGrid = data.frame(k = seq(3, 51, 2)))
fit_knn$bestTune
ggplot(fit_knn)

knn_preds <- predict(fit_knn, test_set)
accuracy_knn <- mean(knn_preds == test_set$class)
fit_knn$results
fit_knn$modelInfo
accuracy_knn


######### Fit KNN Model with Cross-Validation

#set.seed(1991)
set.seed(1991, sample.kind = "Rounding")    # simulate R 3.5
fit_knn_cv <- train(class ~ .,
                      method = "knn",
                      data = train_set,
                      tuneGrid = data.frame(k = seq(3, 51, 2)),
                      trControl = trainControl(method = "cv", number = 10, p = 0.9))
fit_knn_cv$bestTune

knn_cv_preds <- predict(fit_knn_cv, test_set)
accuracy_knn_cv <- mean(knn_cv_preds == test_set$class)
accuracy_knn_cv


######## FINAL TABLE WITH ALL ACCURACIES FROM ALL MODELS AND RESULTS FROM THE BEST KNN MODEL
final_modelaccuracy_table <- data.frame(c(accuracy1, accuracy_rf, accuracy_lda, accuracy_knn, accuracy_knn_cv))
colnames(final_modelaccuracy_table) = c('Classification Tree Model Accuracy', 'Random Forest Accuracy', 'LDA Accuracy', 'KNN Accuracy', 'KNN Accuracy with Cross-Validation')
final_modelaccuracy_table

fit_knn$results


###########################################################################
# 2nd. Task - Classify patients as belonging to one out of two categories: 
#             Normal (100 patients) or Abnormal (210 patients)
###########################################################################

####### Plot of the Counting from each Class
column_2C_weka %>%
  group_by(class) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = class, y = count)) +
  geom_point()


####### Validation set will be 20% of the dataset
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = column_2C_weka$class, times = 1, p = 0.2, list = FALSE)
train_set2 <- column_2C_weka[-test_index,]
test_set2 <- column_2C_weka[test_index,]

#Verification of how many rows in each dataset and how many from each class in order to assure we have enough rows from each class
dim(train_set2)
dim(test_set2)

train_set2 %>%
  group_by(class) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = class, y = count)) +
  geom_point()

test_set2 %>%
  group_by(class) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = class, y = count)) +
  geom_point()

train_set2 %>%
  group_by(class) %>%
  summarize(count = n())

test_set2 %>%
  group_by(class) %>%
  summarize(count = n())

########### Trying Random Guessing

#set.seed(1991)
set.seed(1991, sample.kind = "Rounding")
# guess with equal probability the classes
guess <- sample(c("Abnormal", "Normal"), nrow(test_set), replace = TRUE)
mean(guess == test_set2$class)

########### Fit a Classification Tree Model

x_train2 <- subset(train_set2, select = -c(class))
y_train2 <- train_set2[,7]
x_test2 <- subset(test_set2, select = -c(class))
y_test2 <- test_set2[,7]

# Method 1: Classification Tree Model with the best cp value and standard minsplit (20 observations before splitting)

# set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later

#Training Model
fit_rpart2 <- with(column_2C_weka, 
                  train(x_train2, y_train2, method = "rpart",
                        tuneGrid = data.frame(cp = seq(0, 0.10, 0.01))))

ggplot(fit_rpart2)
confusionMatrix(fit_rpart2)
fit_rpart2$bestTune

#Predicting test_set
p_hat_rpart2 <- predict(fit_rpart2, test_set2)
y_hat_rpart2 <- factor(p_hat_rpart2)

accuracy1_2 <- data.frame("res" = confusionMatrix(y_hat_rpart2, factor(test_set2$class))$overall[["Accuracy"]])
accuracy1_2


# Method 2: Classification Tree Model with the best cp value and minsplit = 0
# set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later

#Training Model
fit_rpart2_2 <- with(column_2C_weka, 
                   train(x_train2, y_train2, method = "rpart",
                         tuneGrid = data.frame(cp = seq(0, 0.10, 0.01)),
                         control = rpart.control(minsplit = 0)))

ggplot(fit_rpart2_2)
confusionMatrix(fit_rpart2_2)
fit_rpart2_2$bestTune

#Predicting test_set
p_hat_rpart2_2 <- predict(fit_rpart2_2, test_set2)
y_hat_rpart2_2 <- factor(p_hat_rpart2_2)

accuracy2_2 <- data.frame("res" = confusionMatrix(y_hat_rpart2_2, factor(test_set2$class))$overall[["Accuracy"]])
accuracy2_2

# make plot of decision tree
plot(fit_rpart2_2$finalModel, margin = 0.1)
text(fit_rpart2_2$finalModel)

### Comparing those Methods the chosen one was the Method 2 - With that the best tune from this method is = 0.04 & Final Accuracy from Classification Tree Model using the test dataset
fit_rpart2_2$bestTune
#Final Accuracy
accuracy2_2

########### Fit a Random Forest Model

# set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later

#Training Model
fit_rf2 <- with(column_2C_weka, 
               train(x_train2, y_train2, method = "rf", 
                     nodesize = 1,
                     tuneGrid = data.frame(mtry = seq(50, 500, 25))))

ggplot(fit_rf2)
confusionMatrix(fit_rf2)
fit_rf2$bestTune

#Predicting test_set
p_hat_rf2 <- predict(fit_rf2, test_set2)
y_hat_rf2 <- factor(p_hat_rf2)

accuracy_rf2 <- data.frame("res" = confusionMatrix(y_hat_rf2, factor(test_set2$class))$overall[["Accuracy"]])
accuracy_rf2

#Variables of Importance
imp2 <- varImp(fit_rf2)
imp2

#Calculating the Variables of Importance from the Random Forest Model

tree_terms <- as.character(unique(fit_rpart2_2$finalModel$frame$var[!(fit_rpart2_2$finalModel$frame$var == "<leaf>")]))
tree_terms

data_frame(term = rownames(imp2$importance), 
           importance = imp2$importance$Overall) %>%
  mutate(rank = rank(-importance)) %>% arrange(desc(importance)) %>%
  filter(term %in% tree_terms)

#Verification if those two main variables are correlated to each other:
cor(column_2C_weka$degree_spondylolisthesis, column_2C_weka$pelvic_radius)
cor(column_2C_weka$degree_spondylolisthesis, column_2C_weka$sacral_slope)
cor(column_2C_weka$pelvic_radius, column_2C_weka$sacral_slope)

########### Fit a LDA Model

#set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later
fit_lda2 <- train(x_train2, y_train2, method = "lda", data = train_set2)
y_hat_lda2 <- predict(fit_lda2, test_set2)
accuracy_lda2 <- mean(y_hat_lda2 == test_set2$class)

########### Fit a Linear Regression Model

#set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later
fit_glm2 <- train(class ~ ., method = "glm", data = train_set2)
glm_preds2 <- predict(fit_glm2, test_set2)
accuracy_glm <- mean(glm_preds2 == test_set2$class)
fit_glm2$results
fit_glm2$modelInfo

########## Fit KNN Model

#set.seed(1991) # if using R 3.5 or earlier
set.seed(1991, sample.kind = "Rounding") # if using R 3.6 or later
fit_knn2 <- train(class ~ .,
                 method = "knn",
                 data = train_set2,
                 tuneGrid = data.frame(k = seq(3, 51, 2)))
fit_knn2$bestTune
ggplot(fit_knn2)

knn_preds2 <- predict(fit_knn2, test_set2)
accuracy_knn2 <- mean(knn_preds2 == test_set2$class)
fit_knn2$results
fit_knn2$modelInfo

######### Fit KNN Model with Cross-Validation

#set.seed(1991)
set.seed(1991, sample.kind = "Rounding")    # simulate R 3.5
fit_knn_cv2 <- train(class ~ .,
                    method = "knn",
                    data = train_set2,
                    tuneGrid = data.frame(k = seq(3, 51, 2)),
                    trControl = trainControl(method = "cv", number = 10, p = 0.9))
fit_knn_cv2$bestTune

knn_cv_preds2 <- predict(fit_knn_cv2, test_set2)
accuracy_knn_cv2 <- mean(knn_cv_preds2 == test_set2$class)

######## FINAL TABLE WITH ALL ACCURACIES FROM ALL MODELS AND RESULTS FROM BOTH LINEAR REGRESSION AND KNN MODELS
final_modelaccuracy_table2 <- data.frame(c(accuracy2_2, accuracy_rf2, accuracy_lda2, accuracy_glm, accuracy_knn2, accuracy_knn_cv2))
colnames(final_modelaccuracy_table2) = c('Classification Tree Model Accuracy', 'Random Forest Accuracy', 'LDA Accuracy', 'Linear Regression Accuracy', 'KNN Accuracy', 'KNN Accuracy with Cross-Validation')
final_modelaccuracy_table2

fit_glm2$results
fit_knn2$results
