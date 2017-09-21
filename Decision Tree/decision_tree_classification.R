# Decision Tree

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

# feature Scaling of numeric columns
cols_to_scale= c("satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
                 "time_spend_company", "Work_accident", "promotion_last_5years")
dataset[, cols_to_scale] = scale(dataset[, cols_to_scale])


# package 'corrr' to find correlations among numeric columns 
if (!"corrr" %in% installed.packages()) { install.packages("corrr") }
library(corrr)
# find correlation among numeric columns of dataset using 'pearson' method
cor(dataset[, sapply(dataset, is.numeric)], use = "complete.obs", method = "pearson")

# satisfaction_level last_evaluation number_project average_montly_hours time_spend_company Work_accident promotion_last_5years
# satisfaction_level            1.00000000     0.105021214   -0.142969586         -0.020048113       -0.100866073   0.058697241           0.025605186
# last_evaluation               0.10502121     1.000000000    0.349332589          0.339741800        0.131590722  -0.007104289          -0.008683768
# number_project               -0.14296959     0.349332589    1.000000000          0.417210634        0.196785891  -0.004740548          -0.006063958
# average_montly_hours         -0.02004811     0.339741800    0.417210634          1.000000000        0.127754910  -0.010142888          -0.003544414
# time_spend_company           -0.10086607     0.131590722    0.196785891          0.127754910        1.000000000   0.002120418           0.067432925
# Work_accident                 0.05869724    -0.007104289   -0.004740548         -0.010142888        0.002120418   1.000000000           0.039245435
# promotion_last_5years         0.02560519    -0.008683768   -0.006063958         -0.003544414        0.067432925   0.039245435           1.000000000

# package 'Hmisc' and 'corrplot' to find correlation and plot it 
install.packages("Hmisc", dependencies = TRUE)
library(Hmisc) 
if (!"corrplot" %in% installed.packages()) { install.packages("corrplot") }
library(corrplot)
# chose only numeric columns for correlation
corr = rcorr(scale(dataset[, sapply(dataset, is.numeric)]))
corr_r = as.matrix(corr[[1]]) 
corr_r[,1]
pval = as.matrix(corr[[3]])
# plot all pairs of columns
corrplot(corr_r,method="circle",type="lower",diag=FALSE,tl.col="black",tl.cex=1,tl.offset=0.1,tl.srt=45, 
         title="Correlation among numeric columns of dataset", mar=c(0,0,1,0))


# splitting dataset into training set and test set
if (!"caTools" %in% installed.packages()) { install.packages("caTools") }
library(caTools)
set.seed(123)

# using package 'rpart' for decision tree 
if (!"rpart" %in% installed.packages()) { install.packages("rpart") }
library(rpart)
# package 'caret' for cross-validation and analysis details 
if (!"caret" %in% installed.packages()) { install.packages("caret") }
library(caret) 

# using 5-fold cross validation 
train_control = trainControl(method="cv", number=5, savePredictions = TRUE)

# applying decision tree 'rpart' to make model
model = train(left~., data=dataset, trControl=train_control, method="rpart")

# predicting values using model
predicted = predict(model,dataset)
# to see predicted and observed values
model$pred

# summarizing results with confusion matrix
cm = confusionMatrix(predicted,dataset$left)
print(cm)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 11217  1156
# 1   211  2415
# 
# Accuracy : 0.9089          
# 95% CI : (0.9041, 0.9134)
# No Information Rate : 0.7619          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7236          
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.9815          
#             Specificity : 0.6763          
#          Pos Pred Value : 0.9066          
#          Neg Pred Value : 0.9196          
#              Prevalence : 0.7619          
#          Detection Rate : 0.7478          
#    Detection Prevalence : 0.8249          
#       Balanced Accuracy : 0.8289          
#                                           
#        'Positive' Class : 0  

plot(model, main = "Accuracy of decision tree model with complexity parameter")
