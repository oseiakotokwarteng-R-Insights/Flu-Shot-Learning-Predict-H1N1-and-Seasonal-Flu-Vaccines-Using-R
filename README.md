

## Overview

[![](images/flu-vaccine.jpg)](https://www.drivendata.org/competitions/66/flu-shot-learning/page/210/)

This report presents a comprehensive analysis and predictive modeling pipeline for vaccine uptake, focusing on both seasonal and H1N1 vaccines from [DrivenData Flu Shot Learning Competition](https://www.drivendata.org/competitions/66/flu-shot-learning/page/210/ "DrivenData"). The primary objective is to build a model that accurately predict respondent vaccination behavior. This report is intended for **R programmers and data scientists** familiar with statistical modeling and machine learning, providing them with a reproducible, end-to-end workflow that covers data pre-processing, predictive models building, hyperparameter tuning, and evaluation.<br> Throughout the report, we leverage popular R packages such as `tidyverse` for data manipulation, `caret` for modeling infrastructure, `xgboost` and `randomForest` for machine learning, and `pROC` for model evaluation. The pipeline emphasizes techniques that can enhance predictive performance, including handling of missing values, encoding of categorical variables, generation of interaction features, and optimization of model hyperparameters.<br> This document is designed not only to present results but also to serve as a practical guide for R programmers to replicate and extend analysis, experiment with alternative models, and potentially improve predictive performance toward achieving higher **AUC scores**.

## Installing all Required Packages

The code below ensures that all the R packages required for the analysis and modeling pipeline are installed and loaded automatically. By using a function to handle both installation and loading, the workflow becomes reproducible and robust, meaning anyone running the script won’t encounter errors due to missing packages. It also simplifies the code, avoiding repeated calls to `install.packages()` or `library()`.

The packages chosen cover data manipulation (`tidyverse`), modeling and evaluation (`caret`, `xgboost`, `randomForest`, `glmnet`, `SuperLearner`), and performance metrics (`pROC`).

```{r, echo=TRUE, warning=FALSE, message=FALSE}
# Define a function to install and load required R packages
install_load <- function(packages) {
  
  # Retrieve a list of all currently installed packages on your system
  installed_package_list <- installed.packages()[, "Package"]
  
  # Identify packages from the input list that are NOT yet installed
  missing_package <- packages[!packages %in% installed_package_list]
  
  # If there are any missing packages, install them along with their dependencies
  if(length(missing_package) > 0) {
    install.packages(missing_package, dependencies = TRUE)
  }
  
  # Load all specified packages into the current R session
  # 'invisible' is used to suppress printing output from lapply
  # 'character.only = TRUE' tells library() to interpret package names as strings
  invisible(lapply(packages, library, character.only = TRUE))
}

# Create a vector of packages required for this analysis
# tidyverse: for data manipulation and visualization
# caret: for machine learning workflows, train/test split, and model evaluation
# xgboost: for gradient boosting modeling
# randomForest: for random forest modeling
# pROC: for computing and plotting ROC curves and AUC
# glmnet: for regularized regression (LASSO, Ridge)
# SuperLearner: for ensemble modeling
my_packages <- c("tidyverse", "caret", "xgboost", "randomForest", "pROC", "glmnet", "SuperLearner")

# Call the function to ensure all packages are installed and loaded
install_load(my_packages)

```

## Loading the Datasets

In this step, we load the training and test datasets provided by the DrivenData “Flu Shot Learning” competition. The datasets include:

1.  `training_features` – Contains demographic, health, and behavioral features for each respondent.

2.  `training_labels` – Contains the target variables indicating whether each respondent received the H1N1 or seasonal flu vaccine.

3.  `test_features` – Contains features for respondents in the test set, which we will use to generate predictions for submission.

We then merge the training features with the training labels using the `respondent_id` as the key. This creates a single dataset called data that includes both predictors and outcomes. This merged dataset will be used for feature engineering, model training, and evaluation.

The reason for merging at this stage is to ensure that each row in our dataset has both the explanatory variables (features) and the response variables (labels), which is necessary for supervised learning models like XGBoost, Random Forest, and Logistic Regression.

```{r, echo=TRUE, warning=FALSE, message=FALSE}
# Load the training features dataset from the DrivenData competition URL
# This dataset contains demographic, health, and behavioral variables for each respondent
training_features <- read.csv("https://drivendata-prod.s3.amazonaws.com/data/66/public/training_set_features.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYSN7TAHVS%2F20251111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251111T154753Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a2738e023bdaf520499879a6b413e9a27035f2e827004311e7e3fad054ea68d8")

# Load the training labels dataset from the DrivenData competition URL
# This dataset contains the target variables (H1N1 and seasonal flu vaccine uptake) for each respondent
training_labels <- read.csv("https://drivendata-prod.s3.amazonaws.com/data/66/public/training_set_labels.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYSN7TAHVS%2F20251111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251111T154753Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=06ee84122c9a6bb285fce6c5bb69b0be488770e63502028b5855ab4832e3c219")

# Load the test features dataset from the DrivenData competition URL
# This dataset contains the same predictor variables as the training set, but for respondents without target labels
test_features <- read.csv("https://drivendata-prod.s3.amazonaws.com/data/66/public/test_set_features.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYSN7TAHVS%2F20251111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251111T154753Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=239bbc286abeebd975a15913a6cb046f209a95b8813c6b3a5d5df201d7a1ff4b")

# Merge the training features with the corresponding labels using 'respondent_id' as the key
# This creates a single dataset combining predictors and targets, which is necessary for supervised learning
data <- merge(training_features, training_labels, by = "respondent_id")

```

## Data Check

Before proceeding with model building, we first explore the dataset to understand its structure and quality. Using summary statistics, we examine the distribution of each variable and identify potential anomalies. The dataset dimensions and structure are checked to confirm the number of observations and variable types. We also assess missing values, both column-wise and overall, to identify any gaps that may require handling. These initial checks ensure that the data is well-understood, clean, and ready for feature engineering and model training.

```{r, echo=TRUE, warning=FALSE, message=FALSE}
# Get a summary of each variable in the dataset, including min, max, mean, quartiles for numeric variables,
# and counts for categorical variables. This helps understand variable distributions and detect anomalies.
summary(data)

# Display the dimensions of the dataset: number of rows (observations) and columns (variables)
dim(data)

# Display the structure of the dataset, including variable types (numeric, factor, etc.) and first few values
str(data)

# Check for missing values in each column by summing NA values column-wise
colSums(is.na(data))

# Get the total number of missing values in the entire dataset
sum(is.na(data))

```

The code below is designed to provide a visual exploration of the numerical variables in the dataset. Understanding the distribution and spread of each feature is essential for feature engineering and model building. The box_hist function automatically generates both a boxplot and a histogram for each variable, which helps in identifying outliers, skewness, and overall distribution patterns. Boxplots highlight extreme values or potential outliers, while histograms provide a sense of the frequency distribution of the variable. This type of exploratory data analysis is an important step before modeling.

```{r, echo=TRUE, warning=FALSE, message=FALSE}
# Exclude the first column (usually an ID column) to focus on features
plot <- data[, -1]
plot <- plot[sapply(plot, is.numeric)] #only numeric columns
str(plot)               # Check the structure of the new 'plot' dataset to ensure it's ready for visualization

# Define a function to plot boxplots and histograms for each variable
box_hist <- function(data) {
  for(var in names(data)) {            # Loop over each column name in the dataset
    par(mfrow = c(1,2))                # Set plotting layout to 1 row, 2 columns (side-by-side plots)
    boxplot(data[[var]],               # Create boxplot for the current variable
            main = paste("Box plot of ", var), 
            ylab = var)
    hist(data[[var]],                  # Create histogram for the current variable
         main = paste("Histogram of ", var), 
         xlab = var)
  }
  par(mfrow = c(1,1))                  # Reset plotting layout to default (1 row, 1 column) after loop
}

# Call the function to generate visualizations for all features
box_hist(plot)

```

## Recoding Values

In this section, we recode categorical variables into numeric form. Many machine learning algorithms, including XGBoost, require numeric inputs to perform computations. By converting ordered categorical variables such as employment status, education level, income, and age group into numeric values, we retain their natural ordering while making them compatible with model training. The numeric coding preserves the inherent ranking of categories which can help the model interpret relative differences between levels.

```{r, echo=TRUE, warning=FALSE, message=FALSE}

# See unique values for employment_status
unique(data$employment_status)

# Recode employment_status into numeric, preserving order
data$employment_status <- as.numeric(factor(data$employment_status, 
                                            levels = c("Not in Labor Force",
                                                       "Unemployed",
                                                       "Employed")))

# check unique values
unique(data$education)

# Recode education into numeric, preserving the education hierarchy
data$education <- as.numeric(factor(data$education, 
                                    levels = c("< 12 Years",
                                               "12 Years",
                                               "Some College",
                                               "College Graduate")))

# Check unique values for income_poverty
unique(data$income_poverty)

# Recode income_poverty into numeric, keeping logical order
data$income_poverty <- as.numeric(factor(data$income_poverty, 
                                         levels = c("Below Poverty",
                                                    "<= $75,000, Above Poverty",
                                                    "> $75,000")))

# Check unique values for age_group
unique(data$age_group)

# Recode age_group into numeric, preserving age progression
data$age_group <- as.numeric(factor(data$age_group, 
                                    levels = c("18 - 34 Years",
                                               "35 - 44 Years",
                                               "45 - 54 Years",
                                               "55 - 64 Years",
                                               "65+ Years")))

```

## Checking Numeric and Categorical Variables

In this section, we inspect the types of columns in the dataset, distinguishing between numeric and character variables. Understanding the variable types is important because many machine learning algorithms require numeric input, and character columns often need to be recoded or encoded.

```{r, echo=TRUE, warning=FALSE, message=FALSE}
# Quickly check which columns are numeric using sapply
sapply(data, is.numeric)

# Define a function to return names of numeric columns
numeric_variables <- function(data) {
  numeric_list <- list()  # Initialize empty list to store numeric columns
  
  # Loop through each column in the data
  for(var in names(data)) {
    var_data <- data[[var]]  # Extract column
    
    if(is.numeric(var_data)){  # Check if column is numeric
      numeric_list[[var]] <- var_data  # Store numeric column in list
    }
  }
  return(names(numeric_list))  # Return names of numeric columns
}

# Apply function to data; confirms there are 30 numeric variables
numeric_variables(data) 

# Quickly check which columns are character using sapply
sapply(data, is.character)

# Define a function to return names of character columns
character_variables <- function(data) {
  character_list <- list()  # Initialize empty list to store character columns
  
  # Loop through each column in the data
  for(var in names(data)) {
    var_data <- data[[var]]  # Extract column
    
    if(is.character(var_data)){  # Check if column is character
      character_list[[var]] <- var_data  # Store character column in list
    }
  }
  return(names(character_list))  # Return names of character columns
}

# Apply function to data; confirms there are 8 character variables
character_variables(data) 

```

## Handling NAs with Median Imputation

In real-world datasets, missing values are common, especially for numeric variables. Many machine learning algorithms cannot handle missing values directly, so we need to impute them.

This code defines a function `replace_numeric_variables` that automatically replaces all missing numeric values (`NA`s) with the median of their respective column. The median is chosen over the mean because it is robust to outliers, ensuring that extreme values do not skew the imputation. After defining the function, it is applied to the dataset `data`, producing a cleaned version where numeric missing values are replaced.

```{r, echo=TRUE, warning=FALSE, message=FALSE}
# Function to replace all numeric variables with their median if they contain NA values
replace_numeric_variables <- function(data) {
  # Loop over every column in the data frame
  for(var in names(data)) {
    # Check if the column is numeric and contains any missing values
    if(is.numeric(data[[var]]) & anyNA(data[[var]])) {
      # Compute the median of the column, ignoring NAs
      col_median <- median(data[[var]], na.rm = TRUE)
      
      # Replace NA values in the column with the computed median
      data[[var]][is.na(data[[var]])] <- col_median
    }
  }
  # Return the modified dataset with NAs replaced
  return(data)
}

# Apply the function to the data
data <- replace_numeric_variables(data)

```

## Dropping Character Variables

In many machine learning workflows, models like XGBoost, Random Forest, and most numerical algorithms cannot directly handle character (string) variables. Converting all variables to numeric or factor is required for modeling.<br> In this step, we identify all character columns in the dataset and remove them, leaving only numeric variables for modeling. This simplifies the dataset and ensures compatibility.

```{r, echo=TRUE, warning=FALSE, message=FALSE}
# Identify character columns in the dataset
char_cols <- sapply(data, is.character)

# Check how many character columns exist
sum(char_cols)

# Drop all character columns from the dataset
data <- data[ , !char_cols]

# Inspect the structure of the updated dataset to confirm character columns are removed
str(data)

```

## Building XGBoost Model for Seasonal and H1n1 vaccines Intake Probabilities

The goal of this section is to build a predictive model for `seasonal_vaccine` and `h1n1_vaccine` target variables using the XGBoost algorithm. XGBoost is chosen because it efficiently handles structured, medium-sized data, captures complex nonlinear relationships and feature interactions, provides robust regularization, and directly outputs probabilities suitable for AUC evaluation. Its combination of predictive performance, flexibility, and scalability makes it an excellent choice for modeling vaccination uptake in this dataset.

```{r, echo=TRUE, warning=FALSE, message=FALSE}
# -----------------------------
# Building XGBoost models
# -----------------------------

set.seed(123)  # Set a random seed for reproducibility of results

# -----------------------------
# Seasonal vaccine model
# -----------------------------
target_seasonal_vaccine <- data$seasonal_vaccine 
# Extract the target variable (binary outcome) for seasonal vaccine

seasonal_predictors <- data %>%
  select(-respondent_id, -seasonal_vaccine, -h1n1_vaccine) 
# Select predictor variables by removing ID and both target columns

matrix_seasonal_predictors <- as.matrix(seasonal_predictors)
# Convert the predictors to a numeric matrix required by XGBoost

# Split the data into training and testing sets (80%-20%)
train_index <- createDataPartition(target_seasonal_vaccine, p = 0.8, list = FALSE)
data_train <- matrix_seasonal_predictors[train_index, ] 
data_test <- matrix_seasonal_predictors[-train_index, ]
target_seasonal_vaccine_train <- target_seasonal_vaccine[train_index] 
target_seasonal_vaccine_test <- target_seasonal_vaccine[-train_index] 

# Convert datasets to XGBoost DMatrix format
xgb_train <- xgb.DMatrix(data = data_train, label = target_seasonal_vaccine_train)
xgb_test <- xgb.DMatrix(data = data_test, label = target_seasonal_vaccine_test)

# Define XGBoost parameters
params <- list(
  objective = "binary:logistic",  # Binary classification with logistic regression
  eval_metric = "auc",            # Evaluation metric: Area Under the ROC Curve
  eta = 0.05,                     # Learning rate (shrinkage) to prevent overfitting
  max_depth = 6,                   # Maximum depth of trees (complexity control)
  subsample = 0.8,                 # Fraction of data used per tree (stochastic sampling)
  colsample_bytree = 0.8           # Fraction of features used per tree (regularization)
)

# Train the XGBoost model
xgb_model <- xgb.train(
  params = params,                 # Model parameters
  data = xgb_train,                # Training dataset
  nrounds = 500,                   # Maximum number of boosting rounds
  watchlist = list(train = xgb_train, test = xgb_test), 
  # Monitor performance on training and test sets
  early_stopping_rounds = 30,      # Stop if test metric does not improve for 30 rounds
  verbose = 0                      # Suppress detailed output
)

# Predict on the test set
xgb_pred <- predict(xgb_model, data_test)

# Evaluate model performance using AUC
auc_xgb <- auc(target_seasonal_vaccine_test, xgb_pred)
print(paste("XGB model: ", auc_xgb))

# -----------------------------
# H1N1 vaccine model
# -----------------------------
set.seed(123)  # Reset seed for reproducibility

target_h1n1_vaccine <- data$h1n1_vaccine 
# Extract the target variable for H1N1 vaccine

h1n1_predictors <- data %>%
  select(-respondent_id, -seasonal_vaccine, -h1n1_vaccine) 
# Predictor variables for H1N1 (exclude ID and both targets)

matrix_h1n1_predictors <- as.matrix(h1n1_predictors)
# Convert to matrix for XGBoost

# Split into training and testing sets (80%-20%)
train_index_2 <- createDataPartition(target_h1n1_vaccine, p = 0.8, list = FALSE)
data_train_2 <- matrix_h1n1_predictors[train_index_2, ]
data_test_2 <- matrix_h1n1_predictors[-train_index_2, ]
target_h1n1_vaccine_train <- target_h1n1_vaccine[train_index_2]
target_h1n1_vaccine_test <- target_h1n1_vaccine[-train_index_2]

# Convert to XGBoost DMatrix
xgb_train_2 <- xgb.DMatrix(data = data_train_2, label = target_h1n1_vaccine_train)
xgb_test_2 <- xgb.DMatrix(data = data_test_2, label = target_h1n1_vaccine_test)

# Train XGBoost model for H1N1
xgb_model_2 <- xgb.train(
  params = params,                 
  data = xgb_train_2,              
  nrounds = 500,                   
  watchlist = list(train = xgb_train_2, test = xgb_test_2), 
  early_stopping_rounds = 30,      
  verbose = 0                      
)

# Predict on the test set
xgb_pred_2 <- predict(xgb_model_2, data_test_2)

# Evaluate performance using AUC
auc_xgb_2 <- auc(target_h1n1_vaccine_test, xgb_pred_2)
print(paste("XGB model: ", auc_xgb_2))

```

## Applying the XGBoost Model to the Test Features Dataset

After training the XGBoost models for predicting seasonal and H1N1 vaccine uptake, the next step is to apply these models to the test dataset to generate predictions. This process involves several key steps: data inspection, variable recoding, missing value handling, and prediction generation.

```{r, echo=TRUE, warning=FALSE, message=FALSE}
################################################################################
###################################################################################
# Apply the trained XGBoost models to the test features dataset

# Inspect the structure of the test dataset
str(test_features)  # Shows variable names, types, and example values

# Recode categorical variables into numeric form for model compatibility
str(test_features)  # Check structure before recoding

# View unique values of 'employment_status' to understand categories
unique(test_features$employment_status)

# Convert 'employment_status' to numeric factor with defined order
test_features$employment_status <- as.numeric(
  factor(test_features$employment_status, 
         levels = c("Not in Labor Force", "Unemployed", "Employed"))
)

str(test_features)  # Check structure after recoding 'employment_status'

# View unique values of 'education'
unique(test_features$education)

# Convert 'education' to numeric factor with defined order
test_features$education <- as.numeric(
  factor(test_features$education, 
         levels = c("< 12 Years", "12 Years", "Some College", "College Graduate"))
)

str(test_features)  # Check structure after recoding 'education'

# View unique values of 'income_poverty'
unique(test_features$income_poverty)

# Convert 'income_poverty' to numeric factor with defined order
test_features$income_poverty <- as.numeric(
  factor(test_features$income_poverty, 
         levels = c("Below Poverty", "<= $75,000, Above Poverty", "> $75,000"))
)

str(test_features)  # Check structure after recoding 'income_poverty'

# View unique values of 'age_group'
unique(test_features$age_group)

# Convert 'age_group' to numeric factor with defined order
test_features$age_group <- as.numeric(
  factor(test_features$age_group, 
         levels = c("18 - 34 Years", "35 - 44 Years", "45 - 54 Years", 
                    "55 - 64 Years", "65+ Years"))
)

# Function to replace all numeric variables that have missing values with the median
replace_numeric_variables <- function(data) {
  for(var in names(data)) {
    # Check if variable is numeric and contains any NA values
    if(is.numeric(data[[var]]) & anyNA(data[[var]])) {
      # Calculate median of the column excluding NAs
      col_median <- median(data[[var]], na.rm = TRUE)
      # Replace NA values with the column median
      data[[var]][is.na(data[[var]])] <- col_median
    }
  }
  return(data)  # Return the modified dataset
}

# Apply the median replacement function to the test dataset
test_features <- replace_numeric_variables(test_features)

# Verify that there are no remaining missing values
sum(is.na(test_features))  # Should return 0

#####
# Drop character variables because XGBoost cannot handle them directly

# Identify character columns
char_cols <- sapply(test_features, is.character)

# Count how many character columns exist
sum(char_cols)

# Remove all character columns
test_features <- test_features[ , !char_cols]

# Inspect the structure of the cleaned test dataset
str(test_features)

# Convert test features to numeric matrix for XGBoost prediction
x_test_final <- as.matrix(
  test_features %>% select(-respondent_id)  # Exclude ID column from predictors
)

# Generate predicted probabilities for seasonal vaccine using the trained model
seasonal_probs <- predict(xgb_model, x_test_final)

# Generate predicted probabilities for H1N1 vaccine using the trained model
h1n1_probs <- predict(xgb_model_2, x_test_final)

# Combine predictions with respondent IDs to create submission dataframe
submission <- data.frame(
  respondent_id = test_features$respondent_id,  # ID column
  h1n1_vaccine = h1n1_probs,                    # Predicted H1N1 probabilities
  seasonal_vaccine = seasonal_probs             # Predicted seasonal vaccine probabilities
)

# Save submission file as CSV (no row names)
write.csv(submission, "submissions_2.csv", row.names = FALSE)

```

## Conclusion

In this report, we successfully built and applied a predictive model to estimate the likelihood of individuals receiving the H1N1 and seasonal vaccines. Using XGBoost, a robust gradient boosting algorithm, allowed us to capture complex, non-linear relationships between demographic, socioeconomic, and health-related predictors and vaccination behavior.

We carefully preprocessed both training and test datasets by handling missing values, encoding categorical variables, and converting features into numeric matrices suitable for modeling. The models were evaluated using the AUC metric, indicating strong discriminatory performance. Finally, predictions were generated for the test dataset and compiled into a submission-ready format.

Further improvements could include hyperparameter tuning, feature engineering, and incorporating additional behavioral or temporal data to enhance prediction accuracy.

