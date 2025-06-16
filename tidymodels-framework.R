# ==========================================
# COMPREHENSIVE TIDYMODELS WORKFLOW
# Step-by-Step Implementation with Parallel Computing
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
  
} else {  # Classification
  
  # Logistic Regression
  logistic_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>%
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
  
  model_list <- list(
    logistic_reg = logistic_spec,
    random_forest = rf_spec,
    xgboost = xgb_spec,
    mars = mars_spec,
    svm_rbf = svm_spec,
    neural_net = nnet_spec,
    bagged_tree = bag_spec,
    naive_bayes = nb_spec
  )
}

cat(paste("Created", length(model_list), "model specifications:\n"))
for (name in names(model_list)) {
  cat(paste("-", name, "\n"))
}
cat("\n")

# ==========================================
# STEP 12: WORKFLOW SETS CREATION
# ==========================================

cat("=== WORKFLOW SETS CREATION ===\n")

# Create workflow sets combining recipes and models
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

cat(paste("Created", nrow(all_workflows), "workflow combinations:\n"),
    paste("- Recipes:", length(recipe_list), "\n"),
    paste("- Models:", length(model_list), "\n"),
    paste("- Total workflows:", nrow(all_workflows), "\n\n"))

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
# STEP 16: VISUALIZE RESULTS
# ==========================================

cat("=== RESULTS VISUALIZATION ===\n")

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
  ) %>%
  arrange(ifelse(MAXIMIZE_METRIC, desc(best_performance), best_performance))

cat("\nPerformance by model type:\n")
print(model_comparison)

# Recipe comparison if multiple recipes used
if (FEATURE_ENGINEERING) {
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
}

cat("\n")

# ==========================================
# STEP 17: ENSEMBLE STACKING (OPTIONAL)
# ==========================================

if (STACK_ENSEMBLE) {
  cat("=== ENSEMBLE STACKING ===\n")
  
  # Create stacks
  cat("Creating model stack...\n")
  model_stack <- stacks() %>%
    add_candidates(grid_results)
  
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
  ensemble_members <- collect_parameters(fitted_stack) %>%
    filter(coef > 0) %>%
    arrange(desc(coef))
  print(ensemble_members)
  
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

cat("\n=== WORKFLOW SUMMARY ===\n")

total_time <- Sys.time() - start_time
cat(paste("Total workflow time:", round(total_time, 2), units(total_time), "\n"),
    paste("Best individual model:", best_workflow_id, "\n"),
    paste("Best", PRIMARY_METRIC, ":", round(best_performance, 4), "\n"))

if (!is.null(fitted_stack)) {
  cat(paste("Ensemble members:", nrow(ensemble_members), "\n"))
  cat(paste("Ensemble", PRIMARY_METRIC, ":", round(ensemble_perf, 4), "\n"))
}

cat("\nWorkflow completed successfully!\n")

# Stop parallel cluster
stopCluster(cl)
cat("\nParallel cluster stopped.\n")

# ==========================================
# OPTIONAL: SAVE RESULTS
# ==========================================

# Uncomment to save results
# save(
#   grid_results, final_fit, fitted_stack, test_metrics,
#   file = paste0("modeling_results_", Sys.Date(), ".RData")
# )
# 
# cat("Results saved to modeling_results_", Sys.Date(), ".RData\n")

# ==========================================
# UTILITY CODE FOR FURTHER ANALYSIS
# ==========================================

# Function to examine specific model results
examine_model <- function(workflow_id) {
  grid_results %>%
    extract_workflow_set_result(workflow_id) %>%
    collect_metrics() %>%
    filter(.metric == PRIMARY_METRIC) %>%
    arrange(desc(mean))
}

examine_model("basic_cubist")

# Function to get predictions from best model
get_predictions <- function() {
  collect_predictions(final_fit)
}

get_predictions()

# Function to compare all models
compare_all_models <- function() {
  grid_results %>%
    rank_results() %>%
    filter(.metric == PRIMARY_METRIC) %>%
    select(wflow_id, model, mean, std_err, rank) %>%
    arrange(rank)
}

compare_all_models()

cat("\nUtility functions available:\n",
    "- examine_model(workflow_id'): Detailed results for specific model\n",
    "- get_predictions(): Get test set predictions from best model\n",
    "- compare_all_models(): Compare all model performances\n",
    "\nExample: examine_model('", best_workflow_id, "')\n")
