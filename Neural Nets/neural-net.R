# Neural Network

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

# using package 'h2o' for artificial neuraal network 
if (!"h2o" %in% installed.packages()) { install.packages("h2o") }
library(h2o)

# number of threads for h2o server (-1: automatically)
h2o.init(nthreads = -1)

# building artificial neuraal network  with activation as 'Rectifier' with (5,5) hidden layers 
model = h2o.deeplearning(y = 'left',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 100,
                         train_samples_per_iteration = -2)
# training se scoring history plot by h2o 
plot(model)

# prediction on test data 
predicted = as.data.frame(h2o.predict(model, newdata = as.h2o(test_set[-7])))

# find accuracy using confusion matrix
cm = table(test_set[, 7], predicted[,1])
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# 
# 0    1
# 0 2215   71
# 1   93  621
# 
# Accuracy : 0.9453          
# 95% CI : (0.9366, 0.9532)
# No Information Rate : 0.7693          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.8477          
# Mcnemar's Test P-Value : 0.101           
# 
# Sensitivity : 0.9597          
# Specificity : 0.8974          
# Pos Pred Value : 0.9689          
# Neg Pred Value : 0.8697          
# Prevalence : 0.7693          
# Detection Rate : 0.7383          
# Detection Prevalence : 0.7620          
# Balanced Accuracy : 0.9286          
# 
# 'Positive' Class : 0          

# with activation as 'Rectifier' with (100,100) hidden layers 
model = h2o.deeplearning(y = 'left',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(100,100),
                         epochs = 100,
                         train_samples_per_iteration = -2)
# training se scoring history plot by h2o 
plot(model)

# prediction on test data 
predicted = as.data.frame(h2o.predict(model, newdata = as.h2o(test_set[-7])))

# find accuracy using confusion matrix
cm = table(test_set[, 7], predicted[,1])
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# 
# 0    1
# 0 2231   55
# 1   52  662
# 
# Accuracy : 0.9643          
# 95% CI : (0.9571, 0.9707)
# No Information Rate : 0.761           
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.9018          
# Mcnemar's Test P-Value : 0.8467          
# 
# Sensitivity : 0.9772          
# Specificity : 0.9233          
# Pos Pred Value : 0.9759          
# Neg Pred Value : 0.9272          
# Prevalence : 0.7610          
# Detection Rate : 0.7437          
# Detection Prevalence : 0.7620          
# Balanced Accuracy : 0.9503          
# 
# 'Positive' Class : 0      


# with activation as 'Tanh' with (50,50) hidden layers with 1000 epoch  
model = h2o.deeplearning(y = 'left',
                         training_frame = as.h2o(training_set),
                         activation = 'Tanh',
                         hidden = c(50,50),
                         epochs = 1000,
                         train_samples_per_iteration = -2)
# training se scoring history plot by h2o 
plot(model)

# prediction on test data 
predicted = as.data.frame(h2o.predict(model, newdata = as.h2o(test_set[-7])))

# find accuracy using confusion matrix
cm = table(test_set[, 7], predicted[,1])
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# 
# 0    1
# 0 2245   41
# 1   49  665
# 
# Accuracy : 0.97            
# 95% CI : (0.9633, 0.9758)
# No Information Rate : 0.7647          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.917           
# Mcnemar's Test P-Value : 0.4606          
# 
# Sensitivity : 0.9786          
# Specificity : 0.9419          
# Pos Pred Value : 0.9821          
# Neg Pred Value : 0.9314          
# Prevalence : 0.7647          
# Detection Rate : 0.7483          
# Detection Prevalence : 0.7620          
# Balanced Accuracy : 0.9603          
# 
# 'Positive' Class : 0 

# with activation as 'Tanh' with (50,50) hidden layers with 150 epoch  
model = h2o.deeplearning(y = 'left',
                         training_frame = as.h2o(training_set),
                         activation = 'Tanh',
                         hidden = c(50,50),
                         epochs = 150,
                         train_samples_per_iteration = -2)
# training se scoring history plot by h2o 
plot(model)

# prediction on test data 
predicted = as.data.frame(h2o.predict(model, newdata = as.h2o(test_set[-7])))

# find accuracy using confusion matrix
cm = table(test_set[, 7], predicted[,1])
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# 
# 0    1
# 0 2247   39
# 1   41  673
# 
# Accuracy : 0.9733          
# 95% CI : (0.9669, 0.9788)
# No Information Rate : 0.7627          
# P-Value [Acc > NIR] : <2e-16          
# 
# Kappa : 0.9264          
# Mcnemar's Test P-Value : 0.911           
# 
# Sensitivity : 0.9821          
# Specificity : 0.9452          
# Pos Pred Value : 0.9829          
# Neg Pred Value : 0.9426          
# Prevalence : 0.7627          
# Detection Rate : 0.7490          
# Detection Prevalence : 0.7620          
# Balanced Accuracy : 0.9637          
# 
# 'Positive' Class : 0    


# shuting down h2o server 
h2o.shutdown()
