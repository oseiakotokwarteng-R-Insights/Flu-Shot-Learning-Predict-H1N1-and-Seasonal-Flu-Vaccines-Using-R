#####
# install and load all required packages
install_load <- function(packages) {
  #retrieves list of installed packages
  installed_package_list <- installed.packages()[, "Package"]
  #check missing packages
  missing_package <- packages[!packages %in% installed_package_list]
  
  #if package is missing, install package and all its dependencies
  if(length(missing_package) > 0) {
    install.packages(missing_package, dependencies = TRUE)
  }
  
  #load the package
  invisible(lapply(packages, library, character.only = TRUE))
}

my_packages <- c("tidyverse", "caret", "xgboost", "randomForest", "pROC", "glmnet", "SuperLearner")
install_load(my_packages)

#####
# loading data
training_features <- read.csv("https://drivendata-prod.s3.amazonaws.com/data/66/public/training_set_features.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYSN7TAHVS%2F20251111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251111T154753Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a2738e023bdaf520499879a6b413e9a27035f2e827004311e7e3fad054ea68d8")
training_labels <- read.csv("https://drivendata-prod.s3.amazonaws.com/data/66/public/training_set_labels.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYSN7TAHVS%2F20251111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251111T154753Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=06ee84122c9a6bb285fce6c5bb69b0be488770e63502028b5855ab4832e3c219")
test_features <- read.csv("https://drivendata-prod.s3.amazonaws.com/data/66/public/test_set_features.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYSN7TAHVS%2F20251111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251111T154753Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=239bbc286abeebd975a15913a6cb046f209a95b8813c6b3a5d5df201d7a1ff4b")
data <- merge(training_features, training_labels, by = "respondent_id")

#####
# overview of data
summary(data)
dim(data)
str(data)
colSums(is.na(data))
sum(is.na(data))

#####
# recode variables
str(data)
unique(data$employment_status)
data$employment_status <- as.numeric(factor(data$employment_status, levels = c("Not in Labor Force",
                                                                                 "Unemployed",
                                                                                 "Employed")))

str(data)
unique(data$education)
data$education <- as.numeric(factor(data$education, levels = c("< 12 Years",
                                                                 "12 Years",
                                                                 "Some College",
                                                                 "College Graduate")))

str(data)
unique(data$income_poverty)
data$income_poverty <- as.numeric(factor(data$income_poverty, levels = c("Below Poverty",
                                                                           "<= $75,000, Above Poverty",
                                                                           "> $75,000")))

str(data)
unique(data$age_group)
data$age_group <- as.numeric(factor(data$age_group, levels = c("18 - 34 Years",
                                                                 "35 - 44 Years",
                                                                 "45 - 54 Years",
                                                                 "55 - 64 Years",
                                                                 "65+ Years")))


#####
# checking for nas
colSums(is.na(data))
sum(is.na(data))
colnames(is.na(data)) #there are 38 columns with nas

######
# checking for numeric and character columns
sapply(data, is.numeric)
numeric_variables <- function(data) {
  numeric_list <- list()
  
  for(var in names(data)) {
    var_data <- data[[var]]
    
    if(is.numeric(var_data)){
      numeric_list[[var]] <- var_data
    }
  }
  return(names(numeric_list))
}

numeric_variables(data) #there are 30 numeric variables


sapply(data, is.character)
character_variables <- function(data) {
  character_list <- list()
  
  for(var in names(data)) {
    var_data <- data[[var]]
    
    if(is.character(var_data)){
      character_list[[var]] <- var_data
    }
  }
  return(names(character_list))
}

character_variables(data) #there are 8 character variables

#####
#function to replace all numeric variables with median
replace_numeric_variables <- function(data) {
  for(var in names(data)) {
    #check if variable is numeric and has any na
    if(is.numeric(data[[var]]) & anyNA(data[[var]])) {
      # Calculate the median for the current column, excluding NAs
      col_median <- median(data[[var]], na.rm = TRUE)
      # Replace NA values in the original data frame column with the calculated median
      data[[var]][is.na(data[[var]])] <- col_median
    }
  }
  return(data)
}
data <- replace_numeric_variables_median(data)
sum(is.na(data))


#####
# drop character variables
char_cols <- sapply(data, is.character)
sum(char_cols)
data <- data[ , !char_cols]
str(data)


#####
# create a function to plot variables
plot <- data[, -1]
str(plot)

box_hist <- function(data) {
  for(var in names(data)) {
    par(mfrow = c(1,2))
    boxplot(data[[var]], main = paste("Box plot of ", var), ylab = var)
    hist(data[[var]], main = paste("Histogram of ", var), xlab = var)
  }
  # Reset plotting layout to default after the function finishes
  par(mfrow = c(1,1))
}

box_hist(plot)

####################################################################################
#################################################################################
# building model using xgboost
set.seed(123)
target_seasonal_vaccine <- data$seasonal_vaccine #target variable
seasonal_predictors <- data %>%
  select(-respondent_id, -seasonal_vaccine, -h1n1_vaccine) #predictors
matrix_seasonal_predictors <- as.matrix(seasonal_predictors)

train_index <- createDataPartition(target_seasonal_vaccine, p = 0.8, list = FALSE)
data_train <- matrix_seasonal_predictors[train_index, ]
data_test <- matrix_seasonal_predictors[-train_index, ]
target_seasonal_vaccine_train <- target_seasonal_vaccine[train_index]
target_seasonal_vaccine_test <- target_seasonal_vaccine[-train_index]

#xgboost
xgb_train <- xgb.DMatrix(data = data_train, label = target_seasonal_vaccine_train)
xgb_test <- xgb.DMatrix(data = data_test, label = target_seasonal_vaccine_test)

# parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params,
  data = xgb_train,
  nrounds = 500,
  watchlist = list(train = xgb_train, test = xgb_test),
  early_stopping_rounds = 30,
  verbose = 0
)

xgb_pred <- predict(xgb_model, data_test)
auc_xgb <- auc(target_seasonal_vaccine_test, xgb_pred)
print(paste("XGB model: ", auc_xgb))


#########################################
set.seed(123)
target_h1n1_vaccine <- data$h1n1_vaccine #target variable
h1n1_predictors <- data %>%
  select(-respondent_id, -seasonal_vaccine, -h1n1_vaccine) #predictors
matrix_h1n1_predictors <- as.matrix(h1n1_predictors)

train_index_2 <- createDataPartition(target_h1n1_vaccine, p = 0.8, list = FALSE)
data_train_2 <- matrix_h1n1_predictors[train_index_2, ]
data_test_2 <- matrix_h1n1_predictors[-train_index_2, ]
target_h1n1_vaccine_train <- target_h1n1_vaccine[train_index_2]
target_h1n1_vaccine_test <- target_h1n1_vaccine[-train_index_2]

#xgboost
xgb_train_2 <- xgb.DMatrix(data = data_train_2, label = target_h1n1_vaccine_train)
xgb_test_2 <- xgb.DMatrix(data = data_test_2, label = target_h1n1_vaccine_test)


xgb_model_2 <- xgb.train(
  params = params,
  data = xgb_train_2,
  nrounds = 500,
  watchlist = list(train = xgb_train_2, test = xgb_test_2),
  early_stopping_rounds = 30,
  verbose = 0
)

xgb_pred_2 <- predict(xgb_model_2, data_test_2)
auc_xgb_2 <- auc(target_h1n1_vaccine_test, xgb_pred_2)
print(paste("XGB model: ", auc_xgb_2))

################################################################################
###################################################################################
# apply the model to the test features dataset

str(test_features)
# recode variables
str(test_features)
unique(test_features$employment_status)
test_features$employment_status <- as.numeric(factor(test_features$employment_status, levels = c("Not in Labor Force",
                                                                               "Unemployed",
                                                                               "Employed")))

str(test_features)
unique(test_features$education)
test_features$education <- as.numeric(factor(test_features$education, levels = c("< 12 Years",
                                                               "12 Years",
                                                               "Some College",
                                                               "College Graduate")))

str(test_features)
unique(test_features$income_poverty)
test_features$income_poverty <- as.numeric(factor(test_features$income_poverty, levels = c("Below Poverty",
                                                                         "<= $75,000, Above Poverty",
                                                                         "> $75,000")))

str(test_features)
unique(test_features$age_group)
test_features$age_group <- as.numeric(factor(test_features$age_group, levels = c("18 - 34 Years",
                                                               "35 - 44 Years",
                                                               "45 - 54 Years",
                                                               "55 - 64 Years",
                                                               "65+ Years")))

#function to replace all numeric variables with median
replace_numeric_variables <- function(data) {
  for(var in names(data)) {
    #check if variable is numeric and has any na
    if(is.numeric(data[[var]]) & anyNA(data[[var]])) {
      # Calculate the median for the current column, excluding NAs
      col_median <- median(data[[var]], na.rm = TRUE)
      # Replace NA values in the original data frame column with the calculated median
      data[[var]][is.na(data[[var]])] <- col_median
    }
  }
  return(data)
}
test_features <- replace_numeric_variables_median(test_features)
sum(is.na(test_features))


#####
# drop character variables
char_cols <- sapply(test_features, is.character)
sum(char_cols)
test_features <- test_features[ , !char_cols]
str(test_features)

x_test_final <- as.matrix(test_features
                          %>%
                            select(-respondent_id))

seasonal_probs <- predict(xgb_model, x_test_final)
h1n1_probs <- predict(xgb_model_2, x_test_final)

submission <- data.frame(
  respondent_id = test_features$respondent_id,
  h1n1_vaccine = h1n1_probs,
  seasonal_vaccine = seasonal_probs
)
write.csv(submission, "submissions_2.csv", row.names = FALSE)





























######################################################
# junk code#
# logistic regression modeling

#splitting the dataset

set.seed(1)
sample <- sample(
  c(TRUE, FALSE),
  nrow(data),
  replace = TRUE,
  prob = c(0.7, 0.3)
)
train <- data[sample, ]
test <- data[!sample, ]

#fitting logistic regression model

model_1 <- glm(seasonal_vaccine ~ 
                # h1n1_concern+
                 h1n1_knowledge+
                 # behavioral_antiviral_meds+
                 # behavioral_avoidance+
                 # behavioral_face_mask+
                 behavioral_wash_hands+
                 # behavioral_large_gatherings+
                 # behavioral_outside_home+
                 behavioral_touch_face+
                 doctor_recc_h1n1+
                 doctor_recc_seasonal+
                 chronic_med_condition+
                 # child_under_6_months+
                 health_worker+
                 health_insurance+
                # opinion_h1n1_vacc_effective+
                 opinion_h1n1_risk+
                 opinion_h1n1_sick_from_vacc+
                 opinion_seas_vacc_effective+
                 opinion_seas_risk+
                 opinion_seas_sick_from_vacc+
                 age_group+
                 education+
                 income_poverty+
                 employment_status,
                 # household_adults+
                 # household_children+
                # h1n1_vaccine,
               family = "binomial",
               data = train)
#disable scientific notation for model summary
options(scipen = 999)

summary(model_1)

model_2 <- glm(h1n1_vaccine ~ 
                 h1n1_concern+
                 # h1n1_knowledge+
                # behavioral_antiviral_meds+
                 # behavioral_avoidance+
                 behavioral_face_mask+
                 # behavioral_wash_hands+
                 behavioral_large_gatherings+
                 # behavioral_outside_home+
                 # behavioral_touch_face+
                 doctor_recc_h1n1+
                 doctor_recc_seasonal+
                 # chronic_med_condition+
                # child_under_6_months+
                 health_worker+
                 # health_insurance+
                 opinion_h1n1_vacc_effective+
                 opinion_h1n1_risk+
                 # opinion_h1n1_sick_from_vacc+
                 opinion_seas_vacc_effective,
                 # opinion_seas_risk+
                 # opinion_seas_sick_from_vacc+
                 # age_group+
                 # education+
                 # income_poverty+
                 # employment_status+
                 # household_adults+
                 # household_children+
                # seasonal_vaccine,
               family = "binomial",
               data = train)
#disable scientific notation for model summary
options(scipen = 999)

summary(model_2)

#####
# assessing model fit
# value over 0.2 - 0.4 indicates that the model fits the data well
pR2(model_1)["McFadden"]
pR2(model_2)["McFadden"]

# compare model using the likelihood test ratio
anova(model_1, test = "Chisq")
anova(model_2, test = "Chisq")

# variable importance, that is importance of each predictor
varImp(model_1)
varImp(model_2)

# check to see if multicolinearity is a problem
# vif above 5 indicates severe multicolinearity
vif(model_1)
vif(model_2)

# inspect influential points
#plot(model_1)
# or
influence.measures(model_1)
influence.measures(model_2)

#stepwise selection to help find more parsimonious model
step(model_1, direction = "both")
step(model_2, direction = "both")

# let's predict probabilities of seasonal and h1n1 vaccines in the test dataset
predicted_seasonal_vaccine <- predict(model_1, test, type = "response")
predicted_h1n1_vaccine <- predict(model_2, test, type = "response")

#evaluate predictive performance using roc
roc_seasonal <- roc(test$seasonal_vaccine, predicted_seasonal_vaccine)
auc(roc_seasonal)

roc_h1n1 <- roc(test$h1n1_vaccine, predicted_h1n1_vaccine)
auc(roc_h1n1)

#plot roc curves
plot(roc_seasonal, col = "blue", main = "ROC Seasonal Vaccine")
plot(roc_h1n1, col = "red", add = TRUE)
legend("bottomright", legend = c("Seasonal", "H1N1"),
       col = c("blue", "red"), lwd = 2)


###################################################################
###################################################################
# preprocess test_features data
# recode variables
str(data)
unique(test_features$employment_status)
test_features$employment_status <- as.numeric(factor(test_features$employment_status, levels = c("Not in Labor Force",
                                                                               "Unemployed",
                                                                               "Employed")))

str(data)
unique(test_features$education)
test_features$education <- as.numeric(factor(test_features$education, levels = c("< 12 Years",
                                                               "12 Years",
                                                               "Some College",
                                                               "College Graduate")))

str(data)
unique(test_features$income_poverty)
test_features$income_poverty <- as.numeric(factor(test_features$income_poverty, levels = c("Below Poverty",
                                                                         "<= $75,000, Above Poverty",
                                                                         "> $75,000")))

str(data)
unique(test_features$age_group)
test_features$age_group <- as.numeric(factor(test_features$age_group, levels = c("18 - 34 Years",
                                                               "35 - 44 Years",
                                                               "45 - 54 Years",
                                                               "55 - 64 Years",
                                                               "65+ Years")))


#function to replace all numeric variables with median
replace_numeric_variables <- function(data) {
  for(var in names(data)) {
    #check if variable is numeric and has any na
    if(is.numeric(data[[var]]) & anyNA(data[[var]])) {
      # Calculate the median for the current column, excluding NAs
      col_median <- median(data[[var]], na.rm = TRUE)
      # Replace NA values in the original data frame column with the calculated median
      data[[var]][is.na(data[[var]])] <- col_median
    }
  }
  return(data)
}
test_features <- replace_numeric_variables_median(test_features)
sum(is.na(test_features))


#####
# drop character variables
char_cols <- sapply(test_features, is.character)
sum(char_cols)
test_features <- test_features[ , !char_cols]
str(test_features)


# keep only relevant variables for each model
vars_model1 <- c("respondent_id",
                 "h1n1_knowledge",
                 "behavioral_wash_hands",
                 "behavioral_touch_face",
                 "doctor_recc_h1n1",
                 "doctor_recc_seasonal",
                 "chronic_med_condition",
                 "health_worker",
                 "health_insurance",
                 "opinion_h1n1_risk",
                 "opinion_h1n1_sick_from_vacc",
                 "opinion_seas_vacc_effective",
                 "opinion_seas_risk",
                 "opinion_seas_sick_from_vacc",
                 "age_group",
                 "education",
                 "income_poverty",
                 "employment_status")

vars_model2 <- c("respondent_id",
                 "h1n1_concern",
                 "behavioral_face_mask",
                 "behavioral_large_gatherings",
                 "doctor_recc_h1n1",
                 "doctor_recc_seasonal",
                 "health_worker",
                 "opinion_h1n1_vacc_effective",
                 "opinion_h1n1_risk",
                 "opinion_seas_vacc_effective")

# subset test_features accordingly
test_model1 <- test_features[, vars_model1]
test_model2 <- test_features[, vars_model2]

# predict probabilities for seasonal and h1n1 vaccine uptake
predicted_test_seasonal <- predict(model_1, newdata = test_model1, type = "response")
predicted_test_h1n1 <- predict(model_2, newdata = test_model2, type = "response")

# combine predictions
submission <- data.frame(
  respondent_id = test_features$respondent_id,
  h1n1_vaccine = predicted_test_h1n1,
  seasonal_vaccine = predicted_test_seasonal
)

# preview output
head(submission)
nrow(submission)

write.csv(submission, "submission.csv", row.names = FALSE)

###############################################################################
################################################################################
# improving the model - xgboost to detect interactions and nonlinear relationships
str(data)

data_xgboost <- data
str(data_xgboost)
data_xgboost$h1n1_vaccine <- as.numeric(data_xgboost$h1n1_vaccine)

#preprocessing
data_xgboost <- data_xgboost %>%
  select(-respondent_id, -seasonal_vaccine)
data_xgboost

x <- as.data.frame(data_xgboost)

y <- x$h1n1_vaccine

x <- x %>%
  select(-h1n1_vaccine)
x

# required by xgboost
x_matrix <- as.matrix(x)

set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
x_train <- x_matrix[train_index, ]
x_test <- x_matrix[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# train an xgboost model
xgb_model <- xgboost(
  data = x_train,
  label = y_train,
  nrounds = 200,
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

#predict probabilities
pred_prob <- predict(xgb_model, x_test)

#evaluate roc and auc
roc_obj <- roc(y_test, pred_prob)
auc_val <- auc(roc_obj)
print(auc_val)

################################################################
##############################################################
# let's fine tune hyperparameters

# Convert to DMatrix
dtrain <- xgb.DMatrix(data = x_train, label = y_train)

# Define hyperparameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,             # smaller learning rate for stability
  max_depth = 6,           # default: 6
  min_child_weight = 1,    # default: 1
  subsample = 0.8,         # row sampling
  colsample_bytree = 0.8,  # feature sampling
  gamma = 0
)

# Cross-validation to find best nrounds
cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 1000,           # large number, early stopping will prevent overfitting
  nfold = 5,                # 5-fold CV
  early_stopping_rounds = 50,
  verbose = 1,
  maximize = TRUE
)

best_nrounds <- cv_model$best_iteration
best_nrounds

# train final model using best parameters
xgb_model <- xgboost(
  data = dtrain,
  params = params,
  nrounds = best_nrounds,
  verbose = 1
)

#evaluate auc on the test set
dtest <- xgb.DMatrix(data = x_test, label = y_test)
preds <- predict(xgb_model, dtest)

roc_obj <- roc(y_test, preds)
auc(roc_obj)
plot(roc_obj, col = "blue", main = "XGBoost ROC Curve")

#########################################################################
#######################################################################
# try this 
## prepare the data
# Separate features and target
y <- data$seasonal_vaccine  # target variable
x <- data %>% select(-respondent_id, -seasonal_vaccine, -h1n1_vaccine)  # predictors

# Convert data to matrix for xgboost
x_matrix <- as.matrix(x)

# Train-test split
set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
x_train <- x_matrix[train_index, ]
x_test  <- x_matrix[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]


# train individual models
# xgboost
xgb_train <- xgb.DMatrix(data = x_train, label = y_train)
xgb_test  <- xgb.DMatrix(data = x_test, label = y_test)

# Set parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params,
  data = xgb_train,
  nrounds = 500,
  watchlist = list(train = xgb_train, test = xgb_test),
  early_stopping_rounds = 30,
  verbose = 0
)

# Predict
xgb_pred <- predict(xgb_model, x_test)
auc_xgb <- auc(y_test, xgb_pred)
auc_xgb

#random forests
rf_model <- randomForest(
  x = x_train,
  y = as.factor(y_train),
  ntree = 500,
  mtry = sqrt(ncol(x_train)),
  importance = TRUE
)

rf_pred <- predict(rf_model, x_test, type = "prob")[,2]
auc_rf <- auc(y_test, rf_pred)
auc_rf

#regularized logistic regression
log_model <- cv.glmnet(
  x_train, y_train, 
  family = "binomial", 
  alpha = 0.5  # elastic net: alpha=1 Lasso, alpha=0 Ridge
)

log_pred <- predict(log_model, x_test, type = "response", s = "lambda.min")
auc_log <- auc(y_test, log_pred)
auc_log

# stack models
stack_train <- data.frame(
  xgb = predict(xgb_model, x_train),
  rf  = predict(rf_model, x_train, type="prob")[,2],
  log = predict(log_model, x_train, type="response", s = "lambda.min")
)

stack_test <- data.frame(
  xgb = xgb_pred,
  rf  = rf_pred,
  log = log_pred
)

# Meta-model: simple logistic regression on predictions
stack_model <- glm(y_train ~ ., data = stack_train, family = "binomial")
stack_pred  <- predict(stack_model, stack_test, type = "response")
auc_stack <- auc(y_test, stack_pred)
auc_stack

#compare auc
cat("XGBoost AUC:", auc_xgb, "\n")
cat("Random Forest AUC:", auc_rf, "\n")
cat("Logistic Regression AUC:", auc_log, "\n")
cat("Stacked Ensemble AUC:", auc_stack, "\n")

#######################################################################
#################################################################
# engineered interactions with xgboost
# Select numeric/binary features (exclude respondent_id and target)
features <- data[, setdiff(names(data), c("respondent_id", "seasonal_vaccine", "h1n1_vaccine"))]

# Convert to numeric matrix for interaction
features_matrix <- as.matrix(features)

# Create pairwise interactions automatically
interaction_matrix <- model.matrix(~ .^2 - 1, data = features)
# Explanation:
#   .^2 : include all pairwise interactions
#   -1  : remove intercept

# Check dimensions
dim(interaction_matrix)

set.seed(123)
train_index <- createDataPartition(data$seasonal_vaccine, p = 0.8, list = FALSE)

x_train_int <- interaction_matrix[train_index, ]
x_test_int  <- interaction_matrix[-train_index, ]
y_train <- data$seasonal_vaccine[train_index]
y_test  <- data$seasonal_vaccine[-train_index]


dtrain <- xgb.DMatrix(data = x_train_int, label = y_train)
dtest  <- xgb.DMatrix(data = x_test_int, label = y_test)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 6,       # deeper trees capture interactions
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model_int <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 20,
  print_every_n = 50
)


pred_int <- predict(xgb_model_int, dtest)
auc_int <- roc(y_test, pred_int)$auc
cat("XGBoost with interactions AUC:", auc_int, "\n")




###############################################################
####### LOAD REQUIRED PACKAGES ########
my_packages <- c("tidyverse", "caret", "xgboost", "pROC", "Matrix")
installed_packages <- installed.packages()[, "Package"]
missing_packages <- my_packages[!my_packages %in% installed_packages]
if(length(missing_packages) > 0) install.packages(missing_packages)
invisible(lapply(my_packages, library, character.only = TRUE))


####### STEP 1: PREPARE DATA WITH INTERACTIONS ########

# Select numeric/binary columns only (exclude IDs and target)
feature_cols <- setdiff(names(data), c("respondent_id", "seasonal_vaccine", "h1n1_vaccine"))
features <- data[, feature_cols]

# Generate all pairwise interactions automatically
# model.matrix(~ .^2 - 1) generates all main effects + interactions
feature_matrix <- model.matrix(~ .^2 - 1, data = features)

# Target variable
y <- data$seasonal_vaccine  # or h1n1_vaccine, you can repeat pipeline for both


####### STEP 2: TRAIN-TEST SPLIT ########
set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)

x_train <- feature_matrix[train_index, ]
x_test  <- feature_matrix[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# Convert to xgboost DMatrix
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test, label = y_test)


####### STEP 3: HYPERPARAMETER TUNING ########

# Define grid for tuning
tune_grid <- expand.grid(
  max_depth = c(4, 6, 8),
  eta = c(0.01, 0.05, 0.1),
  subsample = c(0.7, 0.8),
  colsample_bytree = c(0.7, 0.8)
)

best_auc <- 0
best_params <- NULL
best_nrounds <- 0

# Simple grid search
for(i in 1:nrow(tune_grid)){
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = tune_grid$max_depth[i],
    eta = tune_grid$eta[i],
    subsample = tune_grid$subsample[i],
    colsample_bytree = tune_grid$colsample_bytree[i]
  )
  
  xgb_cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 500,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  max_auc <- max(xgb_cv$evaluation_log$test_auc_mean)
  if(max_auc > best_auc){
    best_auc <- max_auc
    best_params <- params
    best_nrounds <- xgb_cv$best_iteration
  }
}

cat("Best CV AUC:", best_auc, "\n")
print(best_params)
cat("Best nrounds:", best_nrounds, "\n")


####### STEP 4: TRAIN FINAL MODEL ########
final_model <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_nrounds
)

# Evaluate on test set
pred_test <- predict(final_model, dtest)
auc_test <- roc(y_test, pred_test)$auc
cat("Test AUC:", auc_test, "\n")


####### STEP 5: PREDICT ON TEST_FEATURES ########

# Prepare test_features same way as training data
test_features_copy <- test_features[, feature_cols]

# Generate pairwise interactions
test_matrix <- model.matrix(~ .^2 - 1, data = test_features_copy)

# Convert to DMatrix
dtest_final <- xgb.DMatrix(data = test_matrix)

# Predict probabilities
pred_test_features <- predict(final_model, dtest_final)

# Prepare submission
submission <- data.frame(
  respondent_id = test_features$respondent_id,
  seasonal_vaccine = pred_test_features
)
head(submission)

# Write to CSV
write.csv(submission, "submission_xgb_interactions.csv", row.names = FALSE)





