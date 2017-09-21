# Random Forest

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

# creating Random Forest model on training set
if (!"randomForest" %in% installed.packages()) { install.packages("randomForest") }
library(randomForest)
classifier = randomForest(x = training_set[-7],
                          y = training_set$left,
                          ntree = 300)

# predicting the test set with classifier
predicted = predict(classifier, newdata = test_set[-7])

# making Confusion Matrix and finding accuracy of prediction
if (!"caret" %in% installed.packages()) { install.packages("caret") }
library(caret) 
cm = table(test_set[, 7], predicted)
confusionMatrix(cm)
print(cm)

# Confusion Matrix and Statistics
# 
# y_pred
# 0    1
# 0 2283    3
# 1   50  664
# 
# Accuracy : 0.9823         
# 95% CI : (0.977, 0.9867)
# No Information Rate : 0.7777         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.9502         
# Mcnemar's Test P-Value : 2.64e-10       
# 
# Sensitivity : 0.9786         
# Specificity : 0.9955         
# Pos Pred Value : 0.9987         
# Neg Pred Value : 0.9300         
# Prevalence : 0.7777         
# Detection Rate : 0.7610         
# Detection Prevalence : 0.7620         
# Balanced Accuracy : 0.9870         
# 
# 'Positive' Class : 0        

# choosing the number of trees
plot(classifier, main = "Number of trees in Random Forest Classifier")

# find accuracy using manual 5-fold cross validation with random forest
# randomly shuffle dataset
dataset = dataset[sample(nrow(dataset)),]

# create 5 equally size folds for 5-fold cross validation
nfold = 5
folds = cut(seq(1,nrow(dataset)), breaks=nfold, labels=FALSE)

# process each fold one by one
accuracyList = c()
for(i in 1:nfold) {
  # select fold i using which() function 
  testIndexes = which(folds==1, arr.ind=TRUE)
  testData = dataset[testIndexes, ]
  trainData = dataset[-testIndexes, ]
  
  model = randomForest(x = trainData[-7], y = trainData$left, ntree = 300)
  predicted = predict(model, newdata = testData[-7])
  
  cm = table(testData[, 7], predicted)
  accuracy = sum(diag(cm))/sum(cm)*100
  accuracyList = c(accuracyList, accuracy)
}

# print list of accuracies and final average accuracy of 10-fold cross validation
print( accuracyList )
# [1] 98.93333 98.93333 98.96667 98.96667 98.93333
print( mean(accuracyList) )
# [1] 98.94667


# create 10 equally size folds for 10-fold cross validation 
nfold = 10
folds = cut(seq(1,nrow(dataset)), breaks=nfold, labels=FALSE)

# process each fold one by one
accuracyList = c()
for(i in 1:nfold) {
  # select fold i using which() function 
  testIndexes = which(folds==1, arr.ind=TRUE)
  testData = dataset[testIndexes, ]
  trainData = dataset[-testIndexes, ]
  
  model = randomForest(x = trainData[-7], y = trainData$left, ntree = 300)
  predicted = predict(model, newdata = testData[-7])
  
  cm = table(testData[, 7], predicted)
  accuracy = sum(diag(cm))/sum(cm)*100
  accuracyList = c(accuracyList, accuracy)
}

# print list of accuracies and final average accuracy of 10-fold cross validation
print( accuracyList )
# [1] 99.20000 99.13333 99.20000 99.13333 99.13333 99.13333 99.20000 99.13333 99.20000 99.13333
print( mean(accuracyList) )
# [1] 99.16
