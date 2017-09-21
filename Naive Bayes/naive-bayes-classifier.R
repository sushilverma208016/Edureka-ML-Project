# Naive Bayes Classifier

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

# importing library 'e1071' for Naive Bayes
if (!"e1071" %in% installed.packages()) { install.packages("e1071") }
library(e1071)


# naive bayes classifier model
simple_naive_bayes_model = naiveBayes(x = training_set[-7],
                        y = training_set$left)

# Predicting the Test set results
predicted = predict(simple_naive_bayes_model, newdata = test_set[-7])

cm = table(test_set[, 7], predicted)
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# predicted
# 0    1
# 0 1854  432
# 1  208  506
# 
# Accuracy : 0.7867          
# 95% CI : (0.7716, 0.8012)
# No Information Rate : 0.6873          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4691          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Sensitivity : 0.8991          
# Specificity : 0.5394          
# Pos Pred Value : 0.8110          
# Neg Pred Value : 0.7087          
# Prevalence : 0.6873          
# Detection Rate : 0.6180          
# Detection Prevalence : 0.7620          
# Balanced Accuracy : 0.7193          
# 
# 'Positive' Class : 0    



# naive bayes classifier model with laplace value given
naive_bayes_model_with_laplace = naiveBayes(x = training_set[-7],
                                      y = training_set$left, laplace=2000.5, eps=0.1, threshold=0.5)

# Predicting the Test set results
predicted = predict(naive_bayes_model_with_laplace, newdata = test_set[-7])

cm = table(test_set[, 7], predicted)
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# predicted
# 0    1
# 0 1899  387
# 1  223  491
# 
# Accuracy : 0.7967          
# 95% CI : (0.7818, 0.8109)
# No Information Rate : 0.7073          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4804          
# Mcnemar's Test P-Value : 4.121e-11       
#                                           
#             Sensitivity : 0.8949          
#             Specificity : 0.5592          
#          Pos Pred Value : 0.8307          
#          Neg Pred Value : 0.6877          
#              Prevalence : 0.7073          
#          Detection Rate : 0.6330          
#    Detection Prevalence : 0.7620          
#       Balanced Accuracy : 0.7271          
#                                           
#        'Positive' Class : 0 

