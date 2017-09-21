# K-Nearest Neighbor Classifier

# setting current working directory programatically using 'rstudioapi'
if (!"rstudioapi" %in% installed.packages()) { install.packages("rstudioapi") }
library(rstudioapi)
current_path = getActiveDocumentContext()$path 
setwd(dirname(current_path ))

# importing dataset
dataset = read.csv('project_dataset.csv')
# finding levels of categorical columns
sapply(dataset, levels)

# target is 7th column
# factoring 'left' column
dataset$left = factor(dataset$left, levels = c(0, 1))
# factoring 'salary' column
dataset$salary = factor(dataset$salary,
                        levels = c('low', 'medium', 'high'),
                        labels = c(1, 2, 3))
# factoring 'department' column
dataset$department = factor(dataset$department,
                            levels = c('accounting', 'hr', 'IT', 'management', 'marketing', 
                                       'product_mng', 'RandD', 'sales', 'support', 'technical'),
                            labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

# splitting dataset into training set and test set
if (!"caTools" %in% installed.packages()) { install.packages("caTools") }
library(caTools)
set.seed(123)
split = sample.split(dataset$left, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# feature Scaling of numeric columns
cols_to_scale= c("satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
                 "time_spend_company", "Work_accident", "promotion_last_5years")
training_set[, cols_to_scale] = scale(training_set[, cols_to_scale])
test_set[, cols_to_scale] = scale(test_set[, cols_to_scale])

# library 'caret' for Confusion Matrix and finding accuracy of prediction
if (!"caret" %in% installed.packages()) { install.packages("caret") }
library(caret) 

# using library 'class' for k-nn classifier
if (!"class" %in% installed.packages()) { install.packages("class") }
library(class)
# using k=5 for k-nn algorithm 
predicted = knn(train = training_set[, -7], test = test_set[, -7],
             cl = training_set[, 7], k = 5)

# creating confusion matrix and find accuracy 
cm = table(test_set[, 7], predicted)
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# predicted
# 0    1
# 0 2187   99
# 1   52  662
# 
# Accuracy : 0.9497          
# 95% CI : (0.9412, 0.9572)
# No Information Rate : 0.7463          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8643          
# Mcnemar's Test P-Value : 0.0001815       
# 
# Sensitivity : 0.9768          
# Specificity : 0.8699          
# Pos Pred Value : 0.9567          
# Neg Pred Value : 0.9272          
# Prevalence : 0.7463          
# Detection Rate : 0.7290          
# Detection Prevalence : 0.7620          
# Balanced Accuracy : 0.9233          
# 
# 'Positive' Class : 0      


# finding accuracy for different values of k from 1 to 10 using k-nn 
accuracyList = c()
for (k in 1:10) {
  predicted = knn(train = training_set[, -7], test = test_set[, -7],
                  cl = training_set[, 7], k = k)
  
  cm = table(test_set[, 7], predicted)
  c = confusionMatrix(cm)
  accuracy = c$overall['Accuracy']
  message("Value of K: ", k, " Accuracy: ", accuracy)
  accuracyList = c(accuracyList, accuracy)
}
# Value of K: 1 Accuracy: 0.97
# Value of K: 2 Accuracy: 0.952
# Value of K: 3 Accuracy: 0.954666666666667
# Value of K: 4 Accuracy: 0.952666666666667
# Value of K: 5 Accuracy: 0.949666666666667
# Value of K: 6 Accuracy: 0.949666666666667
# Value of K: 7 Accuracy: 0.952
# Value of K: 8 Accuracy: 0.947666666666667
# Value of K: 9 Accuracy: 0.948333333333333
# Value of K: 10 Accuracy: 0.947666666666667
