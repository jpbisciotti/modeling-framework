# ==========================================
# COMPREHENSIVE TIDYMODELS WORKFLOW
# Step-by-Step Implementation 
# ==========================================

# ==========================================
# STEP 1: SETUP AND LIBRARY LOADING
# ==========================================

# Load required libraries
library(tidymodels)
library(stacks)
library(workflowsets)
library(rules)
library(baguette)
library(finetune)
library(embed)
library(textrecipes)
library(themis)
library(DALEXtra)
library(vip)
library(corrplot)
library(skimr)
library(VIM)
library(parallel)
library(doParallel)
library(foreach)
library(earth)

tidymodels_prefer()

# ==========================================
# STEP 2: PARALLEL COMPUTING SETUP
# ==========================================

# Detect available cores and set up parallel processing
n_cores <- detectCores()
n_cores_use <- max(1, floor(n_cores / 2))  # Use half the available cores

cat("System Information:\n",
    paste("Total CPU cores detected:", n_cores, "\n"),
    paste("Cores to be used for parallel processing:", n_cores_use, "\n"))

# Setup parallel cluster
cl <- makePSOCKcluster(n_cores_use)
registerDoParallel(cl)

# Verify parallel setup
cat(paste("Parallel backend registered with", getDoParWorkers(), "workers\n\n"))

# Note: Remember to stop the cluster at the end with: stopCluster(cl)

# ==========================================
# STEP 3: CONFIGURATION VARIABLES
# ==========================================

# Set your configuration here - modify as needed

# BEFORE ANY RANDOM PROCESS: Random seed for reproducibility
SEED <- 2024

# STEP 4: LOAD AND EXAMINE YOUR DATA
EXCLUDE_VARS <- NULL # Variables to exclude, e.g., c("id", "date")

# STEP 6: OUTCOME TYPE DETECTION
OUTCOME_VAR <- "Sale_Price" # CHANGE THIS # Update for your actual data  
OUTCOME_TYPE <- "auto" # "auto", "regression", "binary", "multiclass"

# STEP 7: DATA SPLITTING
TEST_SPLIT <- 0.2 # Proportion for test set

# STEP 8: CROSS-VALIDATION SETUP
CV_FOLDS <- 8 # Number of CV folds
CV_REPEATS <- 1 # Number of CV repeats

# STEP 9: BASE RECIPE CREATION
RARE_CATEGORY_THRESHOLD <- 0.01       # Collapse rare categories in base recipe
KNN_NEIGHBORS <- 5 # Impute missing numeric in base recipe

# STEP 9: BASE RECIPE CREATION: HANDLE_CLASS_IMBALANCE: Apply SMOTE for classification
HANDLE_CLASS_IMBALANCE <- FALSE 
SMOTE_NEIGHBORS <- 5 # Add SMOTE in base for classification if specified 

# STEP 9: BASE RECIPE CREATION: FEATURE_ENGINEERING: Use advanced feature engineering
FEATURE_ENGINEERING <- FALSE 
MAX_TOKEN_FILTER <- 100 # Top tokens to convert a token variable to be filtered based on frequency
CORR_THRESHOLD <- 0.9 # Threshold of absolute correlation values to potentially remove variables that have large absolute correlations with other variables
POLY_DEGREE <- 2 # Create new columns that are basis expansions of variables using orthogonal polynomials
NS_DEG_FREE <- 3 # Create new columns that are basis expansions of variables using natural splines

# STEP 10A: MODEL-SPECIFIC RECIPES CREATION
CREATE_MODEL_SPECIFIC_RECIPES <- TRUE # Create specialized recipes for each model type

# STEP 13: HYPERPARAMETER TUNING SETUP
USE_RACING <- TRUE # Use racing for efficient tuning

# STEP 14: RUN HYPERPARAMETER TUNING
PARAMETER_COMBINATIONS <- 10 # Number of parameter combinations to try

# STEP 17: ENSEMBLE STACKING (OPTIONAL)
STACK_ENSEMBLE <- TRUE # Create stacked ensemble

# STEP 19: MODEL INTERPRETABILITY
INTERPRET_MODELS <- TRUE # Generate interpretability analysis

set.seed(SEED)

cat("Configuration set:\n",
    paste("- Outcome variable:", OUTCOME_VAR, "\n"),
    paste("- Test split:", TEST_SPLIT, "\n"),
    paste("- CV setup:", CV_FOLDS, "folds x", CV_REPEATS, "repeats\n"),
    paste("- Feature engineering:", FEATURE_ENGINEERING, "\n"),
    paste("- Model-specific recipes:", CREATE_MODEL_SPECIFIC_RECIPES, "\n"),
    paste("- Use racing:", USE_RACING, "\n"),
    paste("- Stack ensemble:", STACK_ENSEMBLE, "\n\n"))

# ==========================================
# STEP 4: LOAD AND EXAMINE YOUR DATA
# ==========================================

# Load your data here - replace with your data loading code
# data <- read_csv("your_data.csv")
# data <- read_rds("your_data.rds")
# data <- your_data_object

# For demonstration, using built-in dataset - REPLACE THIS
data(ames, package = "modeldata")
data <- ames

# STOP: Replace the above with your actual data loading code before proceeding

cat("=== DATA OVERVIEW ===\n",
    paste("Data dimensions:", nrow(data), "x", ncol(data), "\n"),
    paste("Outcome variable:", OUTCOME_VAR, "\n\n"))

# Remove excluded variables if specified
if (!is.null(EXCLUDE_VARS)) {
  data <- data %>% select(-all_of(EXCLUDE_VARS))
  cat("Excluded variables removed\n")
}

# ==========================================
# STEP 5: EXPLORATORY DATA ANALYSIS
# ==========================================

cat("=== EXPLORATORY DATA ANALYSIS ===\n")

# Data summary
cat("Data summary:\n")
print(skim(data))

# Check for missing data
missing_summary <- data %>%
  summarise_all(~sum(is.na(.))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_count") %>%
  filter(missing_count > 0) %>%
  arrange(desc(missing_count))

if (nrow(missing_summary) > 0) {
  cat("\nVariables with missing data:\n")
  print(missing_summary)
  
  # Visualize missing data pattern if there are missing values
  if (any(is.na(data))) {
    VIM::aggr(data, col = c('navyblue','red'), numbers = TRUE, sortVars = TRUE)
  }
} else {
  cat("\nNo missing data detected.\n")
}

# ==========================================
# STEP 6: OUTCOME TYPE DETECTION
# ==========================================

cat("\n=== OUTCOME TYPE DETECTION ===\n")

# Determine outcome type automatically if not specified
if (OUTCOME_TYPE == "auto") {
  outcome_col <- data[[OUTCOME_VAR]]
  
  if (is.numeric(outcome_col)) {
    unique_vals <- length(unique(outcome_col[!is.na(outcome_col)]))
    if (unique_vals <= 10) {
      OUTCOME_TYPE <- if (unique_vals == 2) "binary" else "multiclass"
      data[[OUTCOME_VAR]] <- factor(data[[OUTCOME_VAR]])
      cat("Converted numeric outcome to factor\n")
    } else {
      OUTCOME_TYPE <- "regression"
    }
  } else {
    unique_vals <- length(unique(outcome_col[!is.na(outcome_col)]))
    OUTCOME_TYPE <- if (unique_vals == 2) "binary" else "multiclass"
    data[[OUTCOME_VAR]] <- factor(data[[OUTCOME_VAR]])
  }
}

cat(paste("Detected outcome type:", OUTCOME_TYPE, "\n"))

# Set appropriate metrics based on outcome type
if (OUTCOME_TYPE == "regression") {
  METRICS_SET <- metric_set(rmse, rsq, mae)
  PRIMARY_METRIC <- "rmse"
  MAXIMIZE_METRIC <- FALSE
} else {
  METRICS_SET <- metric_set(roc_auc, accuracy, bal_accuracy, sens, spec)
  PRIMARY_METRIC <- "roc_auc"
  MAXIMIZE_METRIC <- TRUE
}

cat(paste("Primary metric:", PRIMARY_METRIC, "\n\n"))

# ==========================================
# STEP 7: DATA SPLITTING
# ==========================================

cat("=== DATA SPLITTING ===\n")

# Stratified split for classification, regular for regression
if (OUTCOME_TYPE %in% c("binary", "multiclass")) {
  data_split <- initial_split(data, prop = 1 - TEST_SPLIT, strata = all_of(OUTCOME_VAR))
} else {
  data_split <- initial_split(data, prop = 1 - TEST_SPLIT)
}

train_data <- training(data_split)
test_data <- testing(data_split)

{
  cat(paste("Training set:", nrow(train_data), "observations\n"),
      paste("Test set:", nrow(test_data), "observations\n"))
  
  # Check outcome distribution
  if (OUTCOME_TYPE == "regression") {
    cat("\nOutcome distribution in training set:\n")
    print(summary(train_data[[OUTCOME_VAR]]))
  } else {
    cat("\nOutcome distribution in training set:\n")
    print(table(train_data[[OUTCOME_VAR]]))
  }
  
  cat("\n")
}

# ==========================================
# STEP 8: CROSS-VALIDATION SETUP
# ==========================================

cat("=== CROSS-VALIDATION SETUP ===\n")

# Create CV folds with stratification for classification
if (OUTCOME_TYPE %in% c("binary", "multiclass")) {
  cv_folds <- vfold_cv(train_data, v = CV_FOLDS, repeats = CV_REPEATS, strata = all_of(OUTCOME_VAR))
} else {
  cv_folds <- vfold_cv(train_data, v = CV_FOLDS, repeats = CV_REPEATS)
}

cat(paste("Cross-validation setup:", CV_FOLDS, "folds x", CV_REPEATS, "repeats =", nrow(cv_folds), "total resamples\n\n"))

# ==========================================
# STEP 9: BASE RECIPE CREATION
# ==========================================

cat("=== BASE RECIPE CREATION ===\n")

# Create base recipe with minimal preprocessing
base_recipe <- recipe(as.formula(paste(OUTCOME_VAR, "~ .")), data = train_data) %>%
  step_zv(all_predictors()) %>%            # Remove zero-variance predictors
  step_nzv(all_predictors()) %>%           # Remove near-zero-variance predictors
  step_impute_knn(all_numeric_predictors(), neighbors = KNN_NEIGHBORS) %>%  # Impute missing numeric
  step_impute_mode(all_nominal_predictors()) %>%               # Impute missing categorical
  step_other(all_nominal_predictors(), threshold = RARE_CATEGORY_THRESHOLD) %>%   # Collapse rare categories
  step_dummy(all_nominal_predictors()) %>%                     # Create dummy variables
  step_normalize(all_numeric_predictors())                     # Normalize numeric predictors

# Add SMOTE for classification if specified
if (OUTCOME_TYPE %in% c("binary", "multiclass") && HANDLE_CLASS_IMBALANCE) {
  base_recipe <- base_recipe %>%
    step_smote(all_outcomes(), neighbors = SMOTE_NEIGHBORS)
}

{
  cat("Base recipe created with:\n",
      "- Zero and near-zero variance removal\n",
      "- KNN imputation for numeric variables\n",
      "- Mode imputation for categorical variables\n",
      "- Rare category collapsing (1% threshold)\n",
      "- Dummy variable encoding\n",
      "- Numeric normalization\n")
  if (OUTCOME_TYPE %in% c("binary", "multiclass") && HANDLE_CLASS_IMBALANCE) {
    cat("- SMOTE for class imbalance\n")
  }
  cat("\n")
}

# ==========================================
# STEP 10: ADVANCED FEATURE ENGINEERING RECIPE
# ==========================================

if (FEATURE_ENGINEERING) {
  cat("=== ADVANCED FEATURE ENGINEERING RECIPE ===\n")
  
  # Create comprehensive feature engineering recipe
  fe_recipe <- recipe(as.formula(paste(OUTCOME_VAR, "~ .")), data = train_data) %>%
    # Initial cleaning
    step_zv(all_predictors()) %>%
    step_nzv(all_predictors()) %>%
    
    # Missing data handling
    step_impute_knn(all_numeric_predictors(), neighbors = KNN_NEIGHBORS) %>%
    step_impute_mode(all_nominal_predictors()) %>%
    
    # Text processing (if any text variables exist)
    step_tokenize(all_nominal_predictors(), token = "words") %>%
    step_stopwords(all_nominal_predictors()) %>%
    step_tokenfilter(all_nominal_predictors(), max_tokens = MAX_TOKEN_FILTER) %>%
    step_tf(all_nominal_predictors()) %>%
    
    # Categorical processing
    step_other(all_nominal_predictors(), threshold = RARE_CATEGORY_THRESHOLD) %>%
    step_dummy(all_nominal_predictors(), one_hot = FALSE) %>%
    
    # Numeric transformations
    step_YeoJohnson(all_numeric_predictors()) %>%
    
    # Feature creation 
    step_interact(terms = ~ all_numeric_predictors():all_numeric_predictors()) %>%
    step_poly(all_numeric_predictors(), degree = POLY_DEGREE, options = list(raw = FALSE)) %>%
    step_ns(all_numeric_predictors(), deg_free = NS_DEG_FREE) %>%
    
    # Normalization
    step_normalize(all_numeric_predictors()) %>%
    
    # Feature selection
    step_corr(all_numeric_predictors(), threshold = CORR_THRESHOLD) %>%
    step_nzv(all_predictors()) %>%
    
    # Class imbalance handling
    {if (OUTCOME_TYPE %in% c("binary", "multiclass") && HANDLE_CLASS_IMBALANCE) 
      step_smote(., all_outcomes(), neighbors = SMOTE_NEIGHBORS) else .} %>%
    
    # Final cleaning
    step_zv(all_predictors())
  
  cat("Advanced feature engineering recipe created with:\n")
  cat("- All base recipe steps\n")
  cat("- Text processing (tokenization, stopwords, TF)\n")
  cat("- Yeo-Johnson transformations\n")
  cat("- Interaction terms\n")
  cat("- Polynomial features (degree ", POLY_DEGREE, ")\n")
  cat("- Natural splines (", NS_DEG_FREE," degrees of freedom)\n")
  cat("- Correlation filtering (",CORR_THRESHOLD," threshold)\n")
  if (OUTCOME_TYPE %in% c("binary", "multiclass") && HANDLE_CLASS_IMBALANCE) {
    cat("- SMOTE for class imbalance\n")
  }
  cat("\n")
  
} else {
  fe_recipe <- base_recipe
  cat("=== USING BASE RECIPE (Advanced FE disabled) ===\n\n")
}

# ==========================================
# STEP 10A: MODEL-SPECIFIC RECIPES CREATION
# ==========================================

if (CREATE_MODEL_SPECIFIC_RECIPES) {
  cat("=== CREATING MODEL-SPECIFIC RECIPES ===\n")
  
  # 1. MINIMAL RECIPE FOR TREE-BASED MODELS
  # Tree-based models can handle categorical data natively and require minimal preprocessing
  tree_recipe <- recipe(as.formula(paste(OUTCOME_VAR, "~ .")), data = train_data) %>%
    step_zv(all_predictors()) %>%                              # Remove zero-variance predictors
    step_impute_knn(all_numeric_predictors(), neighbors = KNN_NEIGHBORS) %>%  # Impute missing numeric
    step_impute_mode(all_nominal_predictors()) %>%             # Impute missing categorical
    step_other(all_nominal_predictors(), threshold = RARE_CATEGORY_THRESHOLD) %>%  # Handle rare categories
    {if (OUTCOME_TYPE %in% c("binary", "multiclass") && HANDLE_CLASS_IMBALANCE) 
      step_smote(., all_outcomes(), neighbors = SMOTE_NEIGHBORS) else .}       # Optional SMOTE
  
  # 2. LINEAR MODEL RECIPE
  # Linear models need dummy variables and normalization
  linear_recipe <- recipe(as.formula(paste(OUTCOME_VAR, "~ .")), data = train_data) %>%
    step_zv(all_predictors()) %>%                              # Remove zero-variance predictors
    step_nzv(all_predictors()) %>%                             # Remove near-zero-variance predictors
    step_impute_knn(all_numeric_predictors(), neighbors = KNN_NEIGHBORS) %>%  # Impute missing numeric
    step_impute_mode(all_nominal_predictors()) %>%             # Impute missing categorical
    step_other(all_nominal_predictors(), threshold = RARE_CATEGORY_THRESHOLD) %>%  # Handle rare categories
    step_dummy(all_nominal_predictors()) %>%                   # Create dummy variables
    step_normalize(all_numeric_predictors()) %>%               # Normalize for regularization
    step_corr(all_numeric_predictors(), threshold = 0.95) %>%  # Remove highly correlated predictors
    {if (OUTCOME_TYPE %in% c("binary", "multiclass") && HANDLE_CLASS_IMBALANCE) 
      step_smote(., all_outcomes(), neighbors = SMOTE_NEIGHBORS) else .}       # Optional SMOTE
  
  # 3. DISTANCE-BASED MODEL RECIPE (SVM, Neural Networks)
  # These models are very sensitive to scaling and feature selection
  distance_recipe <- recipe(as.formula(paste(OUTCOME_VAR, "~ .")), data = train_data) %>%
    step_zv(all_predictors()) %>%                              # Remove zero-variance predictors
    step_nzv(all_predictors()) %>%                             # Remove near-zero-variance predictors
    step_impute_knn(all_numeric_predictors(), neighbors = KNN_NEIGHBORS) %>%  # Impute missing numeric
    step_impute_mode(all_nominal_predictors()) %>%             # Impute missing categorical
    step_other(all_nominal_predictors(), threshold = RARE_CATEGORY_THRESHOLD) %>%  # Handle rare categories
    step_dummy(all_nominal_predictors()) %>%                   # Create dummy variables
    step_YeoJohnson(all_numeric_predictors()) %>%              # Transform skewed distributions
    step_normalize(all_numeric_predictors()) %>%               # Critical: normalize all features
    step_corr(all_numeric_predictors(), threshold = 0.9) %>%   # Remove correlated features
    step_pca(all_numeric_predictors(), threshold = 0.95) %>%   # Optional: dimension reduction
    {if (OUTCOME_TYPE %in% c("binary", "multiclass") && HANDLE_CLASS_IMBALANCE) 
      step_smote(., all_outcomes(), neighbors = SMOTE_NEIGHBORS) else .}       # Optional SMOTE
  
  # 4. MARS-SPECIFIC RECIPE
  # MARS can handle some nonlinearity but benefits from preprocessing
  mars_recipe <- recipe(as.formula(paste(OUTCOME_VAR, "~ .")), data = train_data) %>%
    step_zv(all_predictors()) %>%                              # Remove zero-variance predictors
    step_nzv(all_predictors()) %>%                             # Remove near-zero-variance predictors
    step_impute_knn(all_numeric_predictors(), neighbors = KNN_NEIGHBORS) %>%  # Impute missing numeric
    step_impute_mode(all_nominal_predictors()) %>%             # Impute missing categorical
    step_other(all_nominal_predictors(), threshold = RARE_CATEGORY_THRESHOLD) %>%  # Handle rare categories
    step_dummy(all_nominal_predictors()) %>%                   # Create dummy variables
    step_normalize(all_numeric_predictors()) %>%               # Normalize numeric predictors
    {if (OUTCOME_TYPE %in% c("binary", "multiclass") && HANDLE_CLASS_IMBALANCE) 
      step_smote(., all_outcomes(), neighbors = SMOTE_NEIGHBORS) else .}       # Optional SMOTE
  
  # 5. CUBIST-SPECIFIC RECIPE (minimal preprocessing like trees but with some optimization)
  cubist_recipe <- recipe(as.formula(paste(OUTCOME_VAR, "~ .")), data = train_data) %>%
    step_zv(all_predictors()) %>%                              # Remove zero-variance predictors
    step_impute_knn(all_numeric_predictors(), neighbors = KNN_NEIGHBORS) %>%  # Impute missing numeric
    step_impute_mode(all_nominal_predictors()) %>%             # Impute missing categorical
    step_other(all_nominal_predictors(), threshold = RARE_CATEGORY_THRESHOLD) %>%  # Handle rare categories
    step_dummy(all_nominal_predictors()) %>%                   # Cubist needs dummy variables
    {if (OUTCOME_TYPE %in% c("binary", "multiclass") && HANDLE_CLASS_IMBALANCE) 
      step_smote(., all_outcomes(), neighbors = SMOTE_NEIGHBORS) else .}       # Optional SMOTE
  
  cat("Model-specific recipes created:\n")
  cat("- tree_recipe: Minimal preprocessing for tree-based models\n")
  cat("- linear_recipe: Full preprocessing with regularization considerations\n") 
  cat("- distance_recipe: Extensive preprocessing for distance-based models\n")
  cat("- mars_recipe: Moderate preprocessing for MARS models\n")
  cat("- cubist_recipe: Rule-based model preprocessing\n\n")
  
} else {
  # Use base recipe for all models if model-specific recipes are disabled
  tree_recipe <- base_recipe
  linear_recipe <- base_recipe  
  distance_recipe <- base_recipe
  mars_recipe <- base_recipe
  cubist_recipe <- base_recipe
  cat("=== USING BASE RECIPE FOR ALL MODELS (Model-specific recipes disabled) ===\n\n")
}

# ==========================================
# STEP 11: MODEL SPECIFICATIONS
# ==========================================

cat("=== MODEL SPECIFICATIONS ===\n")

# Model specifications adapted for outcome type
if (OUTCOME_TYPE == "regression") {
  
  # Regularized linear regression
  linear_spec <- linear_reg(penalty = tune(), mixture = tune()) %>%
    set_engine("glmnet")
  
  # Random Forest
  rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
    set_engine("ranger", importance = "impurity") %>%
    set_mode("regression")
  
  # XGBoost
  xgb_spec <- boost_tree(
    trees = tune(), min_n = tune(), tree_depth = tune(),
    learn_rate = tune(), loss_reduction = tune()
  ) %>%
    set_engine("xgboost") %>%
    set_mode("regression")
  
  # MARS
  mars_spec <- mars(num_terms = tune(), prod_degree = tune()) %>%
    set_engine("earth") %>%
    set_mode("regression")
  
  # SVM with RBF kernel
  svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
    set_engine("kernlab") %>%
    set_mode("regression")
  
  # Neural Network
  nnet_spec <- mlp(
    hidden_units = tune(), penalty = tune(), epochs = tune()
  ) %>%
    set_engine("nnet", MaxNWts = 2600) %>%
    set_mode("regression")
  
  # Bagged Trees
  bag_spec <- bag_tree() %>%
    set_engine("rpart", times = 50) %>%
    set_mode("regression")
  
  # Cubist Rules
  cubist_spec <- cubist_rules(committees = tune(), neighbors = tune()) %>%
    set_engine("Cubist")
  
} else {  # Classification
  
  # Logistic Regression
  linear_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>%
    set_engine("glmnet")
  
  # Random Forest
  rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
    set_engine("ranger", importance = "impurity") %>%
    set_mode("classification")
  
  # XGBoost
  xgb_spec <- boost_tree(
    trees = tune(), min_n = tune(), tree_depth = tune(),
    learn_rate = tune(), loss_reduction = tune()
  ) %>%
    set_engine("xgboost") %>%
    set_mode("classification")
  
  # MARS
  mars_spec <- mars(num_terms = tune(), prod_degree = tune()) %>%
    set_engine("earth") %>%
    set_mode("classification")
  
  # SVM with RBF kernel
  svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
    set_engine("kernlab") %>%
    set_mode("classification")
  
  # Neural Network
  nnet_spec <- mlp(
    hidden_units = tune(), penalty = tune(), epochs = tune()
  ) %>%
    set_engine("nnet", MaxNWts = 2600) %>%
    set_mode("classification")
  
  # Bagged Trees
  bag_spec <- bag_tree() %>%
    set_engine("rpart", times = 50) %>%
    set_mode("classification")
  
  # Naive Bayes
  nb_spec <- naive_Bayes() %>%
    set_engine("naivebayes") %>%
    set_mode("classification")
  
  # Remove cubist for classification (not available)
  cubist_spec <- NULL
}

# Create model list with appropriate models for the outcome type
if (OUTCOME_TYPE == "regression") {
  model_list <- list(
    linear_reg = linear_spec,
    random_forest = rf_spec,
    xgboost = xgb_spec,
    mars = mars_spec,
    svm_rbf = svm_spec,
    neural_net = nnet_spec,
    bagged_tree = bag_spec,
    cubist = cubist_spec
  )
} else {
  model_list <- list(
    logistic_reg = linear_spec,
    random_forest = rf_spec,
    xgboost = xgb_spec,
    mars = mars_spec,
    svm_rbf = svm_spec,
    neural_net = nnet_spec,
    bagged_tree = bag_spec,
    naive_bayes = nb_spec
  )
}

{
  cat(paste("Created", length(model_list), "model specifications:\n"))
  for (name in names(model_list)) {
    cat(paste("-", name, "\n"))
  }
  cat("\n")
}

# ==========================================
# STEP 12: ENHANCED WORKFLOW SETS CREATION WITH MODEL-SPECIFIC RECIPES
# ==========================================

cat("=== ENHANCED WORKFLOW SETS CREATION ===\n")

if (CREATE_MODEL_SPECIFIC_RECIPES) {
  
  # Create optimized recipe list based on model requirements
  if (FEATURE_ENGINEERING) {
    optimized_recipe_list <- list(
      # Tree-based models: minimal preprocessing
      tree_minimal = tree_recipe,
      
      # Linear models: full preprocessing with regularization
      linear_full = linear_recipe,
      
      # Distance-based models: extensive preprocessing
      distance_scaled = distance_recipe,
      
      # MARS: moderate preprocessing
      mars_moderate = mars_recipe,
      
      # Cubist: rule-based preprocessing
      cubist_rules = if("cubist" %in% names(model_list)) cubist_recipe else NULL,
      
      # Advanced feature engineering for comparison
      fe_advanced = fe_recipe,
      
      # Base recipe for comparison
      base_simple = base_recipe
    )
  } else {
    optimized_recipe_list <- list(
      # Tree-based models: minimal preprocessing
      tree_minimal = tree_recipe,
      
      # Linear models: full preprocessing
      linear_full = linear_recipe,
      
      # Distance-based models: extensive preprocessing  
      distance_scaled = distance_recipe,
      
      # MARS: moderate preprocessing
      mars_moderate = mars_recipe,
      
      # Cubist: rule-based preprocessing
      cubist_rules = if("cubist" %in% names(model_list)) cubist_recipe else NULL,
      
      # Base recipe for comparison
      base_simple = base_recipe
    )
  }
  
  # Remove NULL recipes
  optimized_recipe_list <- optimized_recipe_list[!sapply(optimized_recipe_list, is.null)]
  
  # Create optimized model subsets for specific recipes
  tree_models <- model_list[intersect(c("random_forest", "bagged_tree"), names(model_list))]
  linear_models <- model_list[intersect(if(OUTCOME_TYPE == "regression") "linear_reg" else "logistic_reg", names(model_list))]
  distance_models <- model_list[intersect(c("svm_rbf", "neural_net"), names(model_list))]
  mars_models <- model_list[intersect("mars", names(model_list))]
  cubist_models <- if("cubist" %in% names(model_list)) model_list["cubist"] else list()
  
  # Create individual workflow sets for each recipe type
  workflow_sets_list <- list()
  
  # Tree models with minimal preprocessing
  if (length(tree_models) > 0) {
    workflow_sets_list[["tree"]] <- workflow_set(
      preproc = list(tree_minimal = optimized_recipe_list[["tree_minimal"]]),
      models = tree_models,
      cross = TRUE
    )
  }
  
  # Linear models with full preprocessing
  if (length(linear_models) > 0) {
    workflow_sets_list[["linear"]] <- workflow_set(
      preproc = list(linear_full = optimized_recipe_list[["linear_full"]]),
      models = linear_models,
      cross = TRUE
    )
  }
  
  # Distance-based models with extensive preprocessing
  if (length(distance_models) > 0) {
    workflow_sets_list[["distance"]] <- workflow_set(
      preproc = list(distance_scaled = optimized_recipe_list[["distance_scaled"]]),
      models = distance_models,
      cross = TRUE
    )
  }
  
  # MARS models with moderate preprocessing
  if (length(mars_models) > 0) {
    workflow_sets_list[["mars"]] <- workflow_set(
      preproc = list(mars_moderate = optimized_recipe_list[["mars_moderate"]]),
      models = mars_models,
      cross = TRUE
    )
  }
  
  # Cubist models with rule-based preprocessing
  if (length(cubist_models) > 0 && "cubist_rules" %in% names(optimized_recipe_list)) {
    workflow_sets_list[["cubist"]] <- workflow_set(
      preproc = list(cubist_rules = optimized_recipe_list[["cubist_rules"]]),
      models = cubist_models,
      cross = TRUE
    )
  }
  
  # Add comparison workflows using base recipe for all models
  workflow_sets_list[["base_comparison"]] <- workflow_set(
    preproc = list(base_simple = optimized_recipe_list[["base_simple"]]),
    models = model_list,
    cross = TRUE
  )
  
  # Add advanced feature engineering comparison if enabled
  if (FEATURE_ENGINEERING && "fe_advanced" %in% names(optimized_recipe_list)) {
    workflow_sets_list[["fe_comparison"]] <- workflow_set(
      preproc = list(fe_advanced = optimized_recipe_list[["fe_advanced"]]),
      models = model_list,
      cross = TRUE
    )
  }
  
  # Combine all workflow sets while preserving the workflow_set class
  # We'll use do.call(rbind, ...) which preserves the class better than bind_rows
  all_workflows_list <- workflow_sets_list[lengths(workflow_sets_list) > 0]
  
  if (length(all_workflows_list) > 1) {
    # Combine multiple workflow sets
    all_workflows <- do.call(rbind, all_workflows_list)
    
    # Ensure it maintains the workflow_set class
    class(all_workflows) <- c("workflow_set", class(all_workflows))
  } else if (length(all_workflows_list) == 1) {
    all_workflows <- all_workflows_list[[1]]
  } else {
    stop("No valid workflow sets were created")
  }
  
  cat("Enhanced workflow sets created with model-specific preprocessing:\n")
  cat(paste("- Total workflows:", nrow(all_workflows), "\n"))
  cat(paste("- Recipe types:", length(optimized_recipe_list), "\n"))
  cat(paste("- Model types:", length(model_list), "\n"))
  cat(paste("- Workflow set class:", paste(class(all_workflows), collapse = ", "), "\n"))
  
} else {
  # Original workflow creation
  if (FEATURE_ENGINEERING) {
    recipe_list <- list(
      basic = base_recipe,
      engineered = fe_recipe
    )
  } else {
    recipe_list <- list(
      basic = base_recipe
    )
  }
  
  all_workflows <- workflow_set(
    preproc = recipe_list,
    models = model_list,
    cross = TRUE
  )
  
  cat("Standard workflow sets created:\n")
  cat(paste("- Total workflows:", nrow(all_workflows), "\n"))
  cat(paste("- Recipe types:", length(recipe_list), "\n"))
  cat(paste("- Model types:", length(model_list), "\n"))
  cat(paste("- Workflow set class:", paste(class(all_workflows), collapse = ", "), "\n"))
}

cat("\nWorkflow combinations:\n")
print(all_workflows)

# ==========================================
# STEP 13: HYPERPARAMETER TUNING SETUP
# ==========================================

cat("\n=== HYPERPARAMETER TUNING SETUP ===\n")

# Control objects for tuning
if (USE_RACING) {
  race_ctrl <- control_race(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE,
    verbose = TRUE
  )
  cat("Using racing methods for efficient tuning\n")
  cat("Racing will eliminate poor performers early\n")
} else {
  grid_ctrl <- control_grid(
    save_pred = TRUE,
    parallel_over = "everything", 
    save_workflow = TRUE,
    verbose = TRUE
  )
  cat("Using full grid search for tuning\n")
  cat("All parameter combinations will be evaluated\n")
}

cat(paste("Parallel processing enabled with", getDoParWorkers(), "workers\n\n"))

# ==========================================
# STEP 14: RUN HYPERPARAMETER TUNING
# ==========================================

cat("=== RUNNING HYPERPARAMETER TUNING ===\n",
    "This may take several minutes to hours depending on data size...\n\n")

# Record start time
start_time <- Sys.time()

if (USE_RACING) {
  
  # Racing approach
  grid_results <- all_workflows %>%
    workflow_map(
      "tune_race_anova",
      seed = SEED,
      resamples = cv_folds,
      grid = PARAMETER_COMBINATIONS,                    # Number of parameter combinations to try
      control = race_ctrl,
      metrics = METRICS_SET,
      verbose = TRUE
    )
  
} else {
  
  # Grid search approach
  grid_results <- all_workflows %>%
    workflow_map(
      "tune_grid",
      seed = SEED,
      resamples = cv_folds,
      grid = PARAMETER_COMBINATIONS,
      control = grid_ctrl,
      metrics = METRICS_SET,
      verbose = TRUE
    )
}

# Record end time
end_time <- Sys.time()
tuning_time <- end_time - start_time

cat("\n=== TUNING COMPLETED ===\n",
    paste("Total tuning time:", round(tuning_time, 2), units(tuning_time), "\n"),
    paste("Workflows completed:", nrow(grid_results), "\n\n"))

# ==========================================
# STEP 15: MODEL EVALUATION AND RANKING
# ==========================================

cat("=== MODEL EVALUATION AND RANKING ===\n")

grid_results_unfiltered <- grid_results

grid_results <- grid_results %>%
  # drop empty result
  dplyr::filter(({.} %>% 
                   dplyr::select(result) %>% 
                   dplyr::pull(result) %>% 
                   purrr::map(length) > 0))

# Collect and rank results
results_summary <- grid_results %>% 
  rank_results(rank_metric = PRIMARY_METRIC) %>%
  filter(.metric == PRIMARY_METRIC)

cat(paste("Top 15 model configurations (ranked by", PRIMARY_METRIC, "):\n"))
print(results_summary %>% slice_head(n = 15))

# Show the best performing workflow
best_workflow_id <- results_summary$wflow_id[1]
best_model_type <- results_summary$model[1]
best_performance <- results_summary$mean[1]

cat(paste("\nBest performing model:", best_workflow_id, "\n"),
    paste("Model type:", best_model_type, "\n"),
    paste("Best", PRIMARY_METRIC, ":", round(best_performance, 4), "\n\n"))

# ==========================================
# STEP 16: ENHANCED RESULTS VISUALIZATION
# ==========================================

cat("=== ENHANCED RESULTS VISUALIZATION ===\n")

# Plot model comparison
p1 <- autoplot(grid_results, metric = PRIMARY_METRIC) +
  labs(title = paste("Model Performance Comparison -", PRIMARY_METRIC))
print(p1)

# Performance by model type
model_comparison <- results_summary %>%
  group_by(model) %>%
  summarise(
    mean_performance = mean(mean),
    best_performance = ifelse(MAXIMIZE_METRIC, max(mean), min(mean)),
    worst_performance = ifelse(MAXIMIZE_METRIC, min(mean), max(mean)),
    n_configs = n(),
    .groups = "drop"
  ) 

if (MAXIMIZE_METRIC) {
  model_comparison_best <- model_comparison %>% 
    arrange(desc(best_performance))
} else {
  model_comparison_best <- model_comparison %>% 
    arrange(best_performance)
}

if (MAXIMIZE_METRIC) {
  model_comparison_mean <- model_comparison %>% 
    arrange(desc(mean_performance))
} else {
  model_comparison_mean <- model_comparison %>% 
    arrange(mean_performance)
}

model_comparison <- model_comparison_best

cat("\nPerformance by model type:\n")
print(model_comparison_best)
print(model_comparison_mean)


# Recipe comparison if model-specific recipes used
if (CREATE_MODEL_SPECIFIC_RECIPES) {
  recipe_comparison <- results_summary %>%
    separate(wflow_id, into = c("recipe", "model"), sep = "_", extra = "merge") %>%
    group_by(recipe) %>%
    summarise(
      mean_performance = mean(mean),
      best_performance = ifelse(MAXIMIZE_METRIC, max(mean), min(mean)),
      n_models = n(),
      .groups = "drop"
    ) %>%
    arrange(ifelse(MAXIMIZE_METRIC, desc(mean_performance), mean_performance))
  
  cat("\nPerformance by recipe type:\n")
  print(recipe_comparison)
  
  # Recipe-Model combination analysis
  recipe_model_comparison <- results_summary %>%
    separate(wflow_id, into = c("recipe", "model"), sep = "_", extra = "merge") %>%
    group_by(recipe, model) %>%
    summarise(
      performance = mean(mean),
      .groups = "drop"
    ) %>%
    arrange(ifelse(MAXIMIZE_METRIC, desc(performance), performance))
  
  cat("\nTop recipe-model combinations:\n")
  print(recipe_model_comparison %>% slice_head(n = 10))
}

cat("\n")

# ==========================================
# CONTINUE WITH REMAINING STEPS...
# (Steps 17-21 remain the same as in the original workflow)
# ==========================================

# ==========================================
# STEP 17: ENSEMBLE STACKING (OPTIONAL)
# ==========================================

if (STACK_ENSEMBLE) {
  cat("=== ENSEMBLE STACKING ===\n")
  
  # Create stacks
  cat("Creating model stack...\n")
  model_stack <- stacks() %>%
    add_candidates(grid_results)
  
  model_stack <- stacks() %>%
    add_candidates(grid_results, name = "grid_results")
  
  cat(paste("Model stack created with", ncol(model_stack) - 1, "candidate members\n"))
  
  # Blend predictions with regularization
  cat("Blending predictions with regularized meta-learner...\n")
  set.seed(SEED)
  
  blended_stack <- model_stack %>%
    blend_predictions(
      penalty = 10^seq(-2, -0.5, length = 20),
      metrics = METRICS_SET
    )
  
  cat("Blending completed!\n")
  print(blended_stack)
  
  # Visualize blending results
  p2 <- autoplot(blended_stack)
  print(p2)
  
  p3 <- autoplot(blended_stack, type = "weights") +
    labs(title = "Ensemble Member Weights")
  print(p3)
  
  # Fit member models to full training data
  cat("\nFitting ensemble members to full training data...\n")
  fitted_stack <- blended_stack %>%
    fit_members()
  
  cat("Ensemble fitting completed!\n")
  
  # Show ensemble details
  cat("\nEnsemble composition:\n")
  collect_parameters(fitted_stack, "grid_results") %>%
    filter(coef > 0) %>%
    arrange(desc(coef))
  
} else {
  fitted_stack <- NULL
  cat("=== SKIPPING ENSEMBLE STACKING ===\n\n")
}

# ==========================================
# STEP 18: FINAL MODEL SELECTION AND FITTING
# ==========================================

cat("=== FINAL MODEL SELECTION AND FITTING ===\n")

# Extract best workflow
best_workflow <- grid_results %>%
  extract_workflow(best_workflow_id)

# Get best parameters
best_params <- grid_results %>%
  extract_workflow_set_result(best_workflow_id) %>%
  select_best(metric = PRIMARY_METRIC)

cat("Best hyperparameters:\n")
print(best_params)

# Finalize workflow with best parameters
final_workflow <- best_workflow %>%
  finalize_workflow(best_params)

# Fit to training data and evaluate on test set
cat("\nFitting final model and evaluating on test set...\n")
final_fit <- final_workflow %>%
  last_fit(split = data_split, metrics = METRICS_SET)

# Collect test set metrics
test_metrics <- collect_metrics(final_fit)
cat("\nFinal model test set performance:\n")
print(test_metrics)

# ==========================================
# STEP 19: MODEL INTERPRETABILITY
# ==========================================

if (INTERPRET_MODELS) {
  cat("\n=== MODEL INTERPRETABILITY ===\n")
  
  # Extract fitted model
  final_fitted_model <- extract_fit_parsnip(final_fit)
  
  # Variable importance (if supported by the model)
  if (final_fitted_model$spec$engine %in% c("ranger", "xgboost")) {
    cat("Generating variable importance plot...\n")
    vip_plot <- final_fitted_model %>%
      vip(num_features = 20) +
      labs(title = paste("Variable Importance -", best_model_type))
    print(vip_plot)
  }
  
  # DALEX explanation
  tryCatch({
    cat("Creating DALEX explainer...\n")
    explainer <- explain_tidymodels(
      final_fitted_model,
      data = train_data %>% select(-all_of(OUTCOME_VAR)),
      y = train_data[[OUTCOME_VAR]],
      label = best_model_type,
      verbose = FALSE
    )
    
    # Model performance
    cat("DALEX model performance:\n")
    model_performance_dalex <- model_performance(explainer)
    print(model_performance_dalex)
    
    # Variable importance via DALEX
    cat("DALEX variable importance (this may take a moment)...\n")
    importance_dalex <- model_parts(explainer, 
                                    B = 10,  # Number of permutations
                                    type = "difference")
    plot(importance_dalex, show_boxplots = FALSE)
    
  }, error = function(e) {
    cat("DALEX interpretation failed:", e$message, "\n")
    cat("Continuing without DALEX analysis...\n")
  })
  
  cat("Interpretability analysis completed\n\n")
}

# ==========================================
# STEP 20: ENSEMBLE EVALUATION (IF CREATED)
# ==========================================

if (!is.null(fitted_stack)) {
  cat("=== ENSEMBLE TEST SET EVALUATION ===\n")
  
  # Predict on test set with ensemble
  ensemble_test_pred <- predict(fitted_stack, test_data)
  
  if (OUTCOME_TYPE == "regression") {
    ensemble_test_metrics <- test_data %>%
      bind_cols(ensemble_test_pred) %>%
      metrics(truth = !!sym(OUTCOME_VAR), estimate = .pred)
  } else {
    ensemble_test_pred_prob <- predict(fitted_stack, test_data, type = "prob")
    ensemble_test_metrics <- test_data %>%
      bind_cols(ensemble_test_pred, ensemble_test_pred_prob) %>%
      metrics(truth = !!sym(OUTCOME_VAR), estimate = .pred_class, .pred_yes)
  }
  
  cat("Ensemble model test set performance:\n")
  print(ensemble_test_metrics)
  
  # Compare individual vs ensemble performance
  cat("\n=== PERFORMANCE COMPARISON ===\n")
  
  individual_perf <- test_metrics %>%
    filter(.metric == PRIMARY_METRIC) %>%
    pull(.estimate)
  
  ensemble_perf <- ensemble_test_metrics %>%
    filter(.metric == PRIMARY_METRIC) %>%
    pull(.estimate)
  
  cat(paste("Individual model", PRIMARY_METRIC, ":", round(individual_perf, 4), "\n"))
  cat(paste("Ensemble model", PRIMARY_METRIC, ":", round(ensemble_perf, 4), "\n"))
  
  improvement <- if (MAXIMIZE_METRIC) {
    (ensemble_perf - individual_perf) / individual_perf * 100
  } else {
    (individual_perf - ensemble_perf) / individual_perf * 100
  }
  
  cat(paste("Improvement:", round(improvement, 2), "%\n"))
  
  if (improvement > 0) {
    cat("✓ Ensemble outperformed individual model\n")
  } else {
    cat("⚠ Individual model performed better than ensemble\n")
  }
}

# ==========================================
# STEP 21: SUMMARY AND CLEANUP
# ==========================================

cat("\n=== ENHANCED WORKFLOW SUMMARY ===\n")

total_time <- Sys.time() - start_time
cat(paste("Total workflow time:", round(total_time, 2), units(total_time), "\n"),
    paste("Best individual model:", best_workflow_id, "\n"),
    paste("Best", PRIMARY_METRIC, ":", round(best_performance, 4), "\n"))

if (CREATE_MODEL_SPECIFIC_RECIPES) {
  cat("Enhanced features used:\n")
  cat("- Model-specific preprocessing recipes\n")
  cat("- Optimized recipe-model combinations\n")
}

if (!is.null(fitted_stack)) {
  cat(paste("Ensemble members:", nrow(fitted_stack), "\n"))
  cat(paste("Ensemble", PRIMARY_METRIC, ":", round(ensemble_perf, 4), "\n"))
}

cat("\nEnhanced workflow completed successfully!\n",
    "Key improvements:\n",
    "- Model-specific preprocessing optimizes each algorithm\n",
    "- Tree models use minimal preprocessing\n",
    "- Distance-based models get extensive normalization\n",
    "- Linear models get appropriate regularization preprocessing\n")

# Stop parallel cluster
stopCluster(cl)
cat("\nParallel cluster stopped.\n")

# ==========================================
# OPTIONAL: SAVE RESULTS
# ==========================================

# Uncomment to save results
# save(
#   grid_results, final_fit, fitted_stack, test_metrics,
#   file = paste0("enhanced_modeling_results_", Sys.Date(), ".RData")
# )
# 
# cat("Results saved to enhanced_modeling_results_", Sys.Date(), ".RData\n")

# ==========================================
# ENHANCED UTILITY CODE FOR FURTHER ANALYSIS
# ==========================================

# Function to examine specific model results
examine_model <- function(workflow_id) {
  grid_results %>%
    extract_workflow_set_result(workflow_id) %>%
    collect_metrics() %>%
    filter(.metric == PRIMARY_METRIC) %>%
    arrange(desc(mean))
}

# Function to compare recipes for a specific model
compare_recipes_for_model <- function(model_name) {
  if (CREATE_MODEL_SPECIFIC_RECIPES) {
    results_summary %>%
      separate(wflow_id, into = c("recipe", "model"), sep = "_", extra = "merge") %>%
      filter(model == model_name) %>%
      select(recipe, model, mean, std_err, rank) %>%
      arrange(rank)
  } else {
    cat("Model-specific recipes not enabled. Set CREATE_MODEL_SPECIFIC_RECIPES = TRUE\n")
  }
}

# Function to get predictions from best model
get_predictions <- function() {
  collect_predictions(final_fit)
}

# Function to compare all models
compare_all_models <- function() {
  grid_results %>%
    rank_results() %>%
    filter(.metric == PRIMARY_METRIC) %>%
    select(wflow_id, model, mean, std_err, rank) %>%
    arrange(rank)
}

# Function to show recipe usage summary
show_recipe_usage <- function() {
  if (CREATE_MODEL_SPECIFIC_RECIPES) {
    results_summary %>%
      separate(wflow_id, into = c("recipe", "model"), sep = "_", extra = "merge") %>%
      count(recipe, model) %>%
      arrange(recipe, model)
  } else {
    cat("Model-specific recipes not enabled. Set CREATE_MODEL_SPECIFIC_RECIPES = TRUE\n")
  }
}

if (FALSE) {
  # Detailed results for specific model
  examine_model("base_simple_cubist")
  
  # Compare preprocessing approaches for a model
  compare_recipes_for_model("simple_cubist")
  
  # Get test set predictions from best model
  get_predictions()
  
  # Compare all model performances
  compare_all_models()
  
  # Show which recipes were used with which models
  show_recipe_usage()
}

