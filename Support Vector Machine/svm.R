# Support Vector Machine

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

# importing library 'e1071' for SVM
if (!"e1071" %in% installed.packages()) { install.packages("e1071") }
library(e1071)


# liner kernel SVM classifier
linear_kernel_svm = svm(formula = left ~ ., data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
predicted = predict(linear_kernel_svm, newdata = test_set[-7])

cm = table(test_set[, 7], predicted)
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# predicted
# 0    1
# 0 2147  139
# 1  540  174
# 
# Accuracy : 0.7737          
# 95% CI : (0.7583, 0.7885)
# No Information Rate : 0.8957          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.2267          
# Mcnemar's Test P-Value : <2e-16          
# 
# Sensitivity : 0.7990          
# Specificity : 0.5559          
# Pos Pred Value : 0.9392          
# Neg Pred Value : 0.2437          
# Prevalence : 0.8957          
# Detection Rate : 0.7157          
# Detection Prevalence : 0.7620          
# Balanced Accuracy : 0.6775          
# 
# 'Positive' Class : 0               



# polynomial kernel SVM classifier of degree 4
polynomial_kernel_svm = svm(formula = left ~ ., data = training_set,
                        type = 'C-classification',
                        kernel = 'polynomial', degree=4)

# Predicting the Test set results
predicted = predict(polynomial_kernel_svm, newdata = test_set[-7])

cm = table(test_set[, 7], predicted)
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# predicted
# 0    1
# 0 2222   64
# 1  121  593
# 
# Accuracy : 0.9383          
# 95% CI : (0.9291, 0.9467)
# No Information Rate : 0.781           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8252          
# Mcnemar's Test P-Value : 3.835e-05       
# 
# Sensitivity : 0.9484          
# Specificity : 0.9026          
# Pos Pred Value : 0.9720          
# Neg Pred Value : 0.8305          
# Prevalence : 0.7810          
# Detection Rate : 0.7407          
# Detection Prevalence : 0.7620          
# Balanced Accuracy : 0.9255          
# 
# 'Positive' Class : 0   



# sigmoid kernal SVM classifier
sigmoid_kernel_svm = svm(formula = left ~ ., data = training_set,
                            type = 'C-classification',
                            kernel = 'sigmoid')

# Predicting the Test set results
predicted = predict(sigmoid_kernel_svm, newdata = test_set[-7])

cm = table(test_set[, 7], predicted)
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# predicted
# 0    1
# 0 1677  609
# 1  655   59
# 
# Accuracy : 0.5787          
# 95% CI : (0.5608, 0.5964)
# No Information Rate : 0.7773          
# P-Value [Acc > NIR] : 1.0000          
# 
# Kappa : -0.1879         
# Mcnemar's Test P-Value : 0.2056          
# 
# Sensitivity : 0.71913         
# Specificity : 0.08832         
# Pos Pred Value : 0.73360         
# Neg Pred Value : 0.08263         
# Prevalence : 0.77733         
# Detection Rate : 0.55900         
# Detection Prevalence : 0.76200         
# Balanced Accuracy : 0.40372         
# 
# 'Positive' Class : 0               

