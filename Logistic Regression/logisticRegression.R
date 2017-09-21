# Logistic Regression Classifier

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


# using 'glm' to construct logistic regression classifier using 'binomial' family 
model = glm(formula = left ~ ., family = "binomial", data = training_set)

# using logistic regression model to predict test data
probPredicted = predict(model, type = 'response', newdata = test_set[-7])
predicted = ifelse(probPredicted > 0.5, 1, 0)

# creating confusion matrix and find accuracy 
cm = table(test_set[, 7], predicted)
confusionMatrix(cm)
# Confusion Matrix and Statistics
# 
# predicted
# 0    1
# 0 2117  169
# 1  458  256
# 
# Accuracy : 0.791          
# 95% CI : (0.776, 0.8054)
# No Information Rate : 0.8583         
# P-Value [Acc > NIR] : 1              
# 
# Kappa : 0.3306         
# Mcnemar's Test P-Value : <2e-16         
# 
# Sensitivity : 0.8221         
# Specificity : 0.6024         
# Pos Pred Value : 0.9261         
# Neg Pred Value : 0.3585         
# Prevalence : 0.8583         
# Detection Rate : 0.7057         
# Detection Prevalence : 0.7620         
# Balanced Accuracy : 0.7122         
# 
# 'Positive' Class : 0              

# different families for error distribution function in loistic regression 
familyList = c("binomial", "gaussian", "Gamma", "inverse.gaussian", "poisson", "quasi", "quasibinomial", "quasipoisson") 

for (f in familyList) {
  model = glm(formula = left ~ ., family = binomial, data = training_set)
  
  # using logistic regression model to predict test data
  probPredicted = predict(model, type = 'response', newdata = test_set[-7])
  predicted = ifelse(probPredicted > 0.4, 1, 0)
  
  # creating confusion matrix and find accuracy 
  cm = table(test_set[, 7], predicted)
  c = confusionMatrix(cm)
  accuracy = c$overall['Accuracy']
  message("Family : ", f, " Accuracy: ", accuracy)
}
# Family : binomial Accuracy: 0.793333333333333
# Family : gaussian Accuracy: 0.793333333333333
# Family : Gamma Accuracy: 0.793333333333333
# Family : inverse.gaussian Accuracy: 0.793333333333333
# Family : poisson Accuracy: 0.793333333333333
# Family : quasi Accuracy: 0.793333333333333
# Family : quasibinomial Accuracy: 0.793333333333333
# Family : quasipoisson Accuracy: 0.793333333333333
