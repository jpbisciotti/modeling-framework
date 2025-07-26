# ==============================================================================
# Supervised Feature Selection Methods 
# Four minimally reproducible examples demonstrating different approaches
# ==============================================================================

# Load required packages
required_packages <- c("glmnet", "rpart", "earth", "MASS")
install_if_missing <- function(packages) {
  missing_packages <- packages[!packages %in% installed.packages()[,"Package"]]
  if(length(missing_packages)) install.packages(missing_packages)
}
install_if_missing(required_packages)

library(glmnet)
library(rpart)
library(earth)
library(MASS)

# ==============================================================================
# DATA GENERATION
# Create synthetic datasets with diverse distributions and interaction effects
# ==============================================================================

generate_feature_selection_data <- function(n_samples = 5000, 
                                            n_informative = 5, 
                                            n_noise = 20,
                                            distribution_type = "mixed",
                                            include_interactions = TRUE,
                                            noise_level = 0.5) {
  
  set.seed(42)  # Ensure reproducibility
  
  # Generate informative predictors from specified distribution
  X_informative <- switch(distribution_type,
                          "normal" = {
                            # Standard normal with different means and variances for diversity
                            X <- matrix(nrow = n_samples, ncol = n_informative)
                            for (i in 1:n_informative) {
                              mean_val <- runif(1, -10, 10)  # Random mean
                              sd_val <- runif(1, 0.5, 15) # Random standard deviation
                              X[, i] <- rnorm(n_samples, mean = mean_val, sd = sd_val)
                            }
                            X
                          },
                          
                          "poisson" = {
                            # Poisson with different rate parameters (good for count data)
                            X <- matrix(nrow = n_samples, ncol = n_informative)
                            for (i in 1:n_informative) {
                              lambda_val <- runif(1, 2, 8)  # Rate parameter between 2 and 8
                              X[, i] <- rpois(n_samples, lambda = lambda_val)
                            }
                            scale(X)  # Standardize for consistent interpretation
                          },
                          
                          "binomial" = {
                            # Binomial with different success probabilities
                            X <- matrix(nrow = n_samples, ncol = n_informative)
                            for (i in 1:n_informative) {
                              prob_val <- runif(1, 0.3, 0.7)  # Success probability
                              size_val <- sample(c(10, 20), 1)  # Number of trials
                              X[, i] <- rbinom(n_samples, size = size_val, prob = prob_val)
                            }
                            scale(X)  # Standardize for consistent scale
                          },
                          
                          "exponential" = {
                            # Exponential distribution (good for time-to-event data)
                            X <- matrix(nrow = n_samples, ncol = n_informative)
                            for (i in 1:n_informative) {
                              rate_val <- runif(1, 0.5, 10)  # Rate parameter
                              X[, i] <- rexp(n_samples, rate = rate_val)
                            }
                            scale(log(X + 0.1))  # Log transform and standardize
                          },
                          
                          "gamma" = {
                            # Gamma distribution (good for positive continuous data like income)
                            X <- matrix(nrow = n_samples, ncol = n_informative)
                            for (i in 1:n_informative) {
                              shape_val <- runif(1, 1, 10)   # Shape parameter
                              rate_val <- runif(1, 0.5, 15)  # Rate parameter
                              X[, i] <- rgamma(n_samples, shape = shape_val, rate = rate_val)
                            }
                            scale(X)  # Standardize for consistent interpretation
                          },
                          
                          "mixed" = {
                            # Mixture of distributions for real-world diversity
                            X <- matrix(nrow = n_samples, ncol = n_informative)
                            distributions <- c("normal", "poisson", "exponential", "gamma")
                            
                            for (i in 1:n_informative) {
                              dist_choice <- sample(distributions, 1)
                              temp_data <- switch(dist_choice,
                                                  "normal" = rnorm(n_samples, mean = runif(1, -10, 10), sd = runif(1, 0.5, 15)),
                                                  "poisson" = scale(rpois(n_samples, lambda = runif(1, 2, 8)))[, 1],
                                                  "exponential" = scale(log(rexp(n_samples, rate = runif(1, 0.5, 10)) + 0.1))[, 1],
                                                  "gamma" = scale(rgamma(n_samples, shape = runif(1, 1, 10), rate = runif(1, 0.5, 15)))[, 1]
                              )
                              X[, i] <- temp_data
                            }
                            X
                          },
                          
                          # Default to normal if unrecognized distribution
                          {
                            warning(paste("Unknown distribution:", distribution_type, ". Using normal."))
                            matrix(rnorm(n_samples * n_informative), ncol = n_informative)
                          }
  )
  
  colnames(X_informative) <- paste0("signal_", 1:n_informative)
  
  # Generate noise predictors from mixed distributions to test robustness
  X_noise <- matrix(nrow = n_samples, ncol = n_noise)
  for (i in 1:n_noise) {
    dist_choice <- sample(c("normal", "uniform", "poisson", "exponential", "gamma"), 1)
    X_noise[, i] <- switch(dist_choice,
                           "normal" = rnorm(n_samples, mean = runif(1, -10, 10), sd = runif(1, 0.5, 15)),
                           "poisson" = scale(rpois(n_samples, lambda = runif(1, 2, 8)))[, 1],
                           "exponential" = scale(log(rexp(n_samples, rate = runif(1, 0.5, 10)) + 0.1))[, 1],
                           "gamma" = scale(rgamma(n_samples, shape = runif(1, 1, 10), rate = runif(1, 0.5, 15)))[, 1],
                           "uniform" = runif(n_samples, min = -20, max = 20)
    )
  }
  colnames(X_noise) <- paste0("noise_", 1:n_noise)
  
  # Combine all predictors
  X <- cbind(X_informative, X_noise)
  
  # Generate response with main effects
  main_coefficients <- round(runif(n = n_informative, min = -10, max = 10), 1)
  
  y_main <- X_informative %*% main_coefficients
  
  # Add interaction effects if requested
  y_interactions <- 0
  interaction_features <- character(0)
  
  if (include_interactions && n_informative >= 2) {
    # Add pairwise interaction between first two features
    interaction_coef <- 0.5
    y_interactions <- y_interactions + interaction_coef * X_informative[, 1] * X_informative[, 2]
    interaction_features <- c("signal_1", "signal_2")
    
    # Add quadratic effect for first feature to test non-linear detection
    if (n_informative >= 1) {
      quad_coef <- 0.3
      y_interactions <- y_interactions + quad_coef * X_informative[, 1]^2
    }
  }
  
  # Add noise to response
  y_noise <- rnorm(n_samples, mean = 0, sd = noise_level)
  
  # Combine all components
  y <- as.vector(y_main + y_interactions + y_noise)
  
  return(list(
    X = X,
    y = y,
    true_features = colnames(X_informative),
    interaction_features = interaction_features,
    noise_features = colnames(X_noise),
    distribution_type = distribution_type,
    include_interactions = include_interactions
  ))
}

# Generate shared dataset for all methods
data <- generate_feature_selection_data(
  n_samples = 10000,
  n_informative = 10, 
  n_noise = 50,
  distribution_type = "mixed",  # Test with mixed distributions
  include_interactions = TRUE,
  noise_level = 0.5
)

X <- data$X
y <- data$y

cat("Dataset created:\n",
    "- Distribution type:", data$distribution_type, "\n",
    "- Total features:", ncol(X), "\n",
    "- True signal features:", length(data$true_features), "\n",
    "- Features with interactions:", length(data$interaction_features), "\n",
    "- Noise features:", length(data$noise_features), "\n",
    "- Sample size:", length(y), "\n\n")

# ==============================================================================
# METHOD 1: LASSO AND REGULARIZATION
# Uses L1 penalty to drive coefficients to zero for automatic feature selection
# ==============================================================================

perform_lasso_selection <- function(X, y) {
  cat("=== LASSO Feature Selection ===\n")
  
  # Fit LASSO with cross-validation to find optimal lambda
  cv_lasso <- cv.glmnet(X, y, alpha = 1, nfolds = 10)
  
  # Extract coefficients at optimal lambda (lambda.1se for more conservative selection)
  optimal_lambda <- cv_lasso$lambda.1se
  lasso_model <- glmnet(X, y, alpha = 1, lambda = optimal_lambda)
  coefficients <- coef(lasso_model, s = optimal_lambda)
  
  # Identify selected features (non-zero coefficients, excluding intercept)
  selected_features <- rownames(coefficients)[coefficients[-1, 1] != 0]
  
  # Calculate selection performance
  true_positives <- sum(selected_features %in% data$true_features)
  false_positives <- sum(selected_features %in% data$noise_features)
  
  cat("Optimal lambda:", round(optimal_lambda, 4), "\n")
  cat("Selected features:", length(selected_features), "\n")
  cat("True positives:", true_positives, "/", length(data$true_features), "\n")
  cat("False positives:", false_positives, "\n")
  cat("Selected features:", paste(selected_features, collapse = ", "), "\n\n")
  
  return(list(
    model = lasso_model,
    selected_features = selected_features,
    lambda = optimal_lambda,
    cv_results = cv_lasso
  ))
}

lasso_results <- perform_lasso_selection(X, y)

# ==============================================================================
# METHOD 2: TREE-BASED FEATURE SELECTION
# Uses variable importance from decision trees for feature ranking
# ==============================================================================

perform_tree_selection <- function(X, y, importance_threshold = 0.01) {
  cat("=== Tree-Based Feature Selection ===\n")
  
  # Create data frame for rpart
  df <- data.frame(y = y, X)
  
  # Fit regression tree with complexity parameter tuning
  tree_model <- rpart(y ~ ., data = df, method = "anova", 
                      control = rpart.control(cp = 0.001, minsplit = 10))
  
  # Prune tree using cross-validation results
  cp_optimal <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]
  pruned_tree <- prune(tree_model, cp = cp_optimal)
  
  # Extract variable importance
  importance_scores <- pruned_tree$variable.importance
  
  # Select features above importance threshold
  if(length(importance_scores) > 0) {
    # Normalize importance scores
    normalized_importance <- importance_scores / sum(importance_scores)
    selected_features <- names(normalized_importance[normalized_importance >= importance_threshold])
  } else {
    selected_features <- character(0)
  }
  
  # Calculate selection performance
  true_positives <- sum(selected_features %in% data$true_features)
  false_positives <- sum(selected_features %in% data$noise_features)
  
  cat("Optimal CP:", round(cp_optimal, 4), "\n")
  cat("Selected features:", length(selected_features), "\n")
  cat("True positives:", true_positives, "/", length(data$true_features), "\n")
  cat("False positives:", false_positives, "\n")
  cat("Selected features:", paste(selected_features, collapse = ", "), "\n")
  
  if(length(importance_scores) > 0) {
    cat("Top 5 important features:\n")
    top_features <- head(sort(normalized_importance, decreasing = TRUE), 5)
    for(i in 1:length(top_features)) {
      cat(sprintf("  %s: %.3f\n", names(top_features)[i], top_features[i]))
    }
  }
  cat("\n")
  
  return(list(
    model = pruned_tree,
    selected_features = selected_features,
    importance_scores = importance_scores,
    threshold = importance_threshold
  ))
}

tree_results <- perform_tree_selection(X, y)

# ==============================================================================
# METHOD 3: INFORMATION CRITERION-BASED SELECTION
# Uses AIC/BIC for stepwise feature selection
# ==============================================================================

perform_ic_selection <- function(X, y, criterion = "AIC") {
  cat("=== Information Criterion-Based Selection (", criterion, ") ===\n", sep = "")
  
  # Create data frame
  df <- data.frame(y = y, X)
  
  # Fit full model
  full_model <- lm(y ~ ., data = df)
  
  # Fit null model (intercept only)
  null_model <- lm(y ~ 1, data = df)
  
  # Perform stepwise selection
  if(criterion == "AIC") {
    step_model <- stepAIC(null_model, 
                          scope = list(lower = null_model, upper = full_model),
                          direction = "forward", 
                          trace = FALSE)
  } else {  # BIC
    step_model <- stepAIC(null_model,
                          scope = list(lower = null_model, upper = full_model),
                          direction = "forward",
                          k = log(nrow(df)),  # BIC penalty
                          trace = FALSE)
  }
  
  # Extract selected features
  selected_features <- names(step_model$coefficients)[-1]  # Remove intercept
  
  # Calculate selection performance
  true_positives <- sum(selected_features %in% data$true_features)
  false_positives <- sum(selected_features %in% data$noise_features)
  
  # Model comparison metrics
  full_criterion <- if(criterion == "AIC") AIC(full_model) else BIC(full_model)
  selected_criterion <- if(criterion == "AIC") AIC(step_model) else BIC(step_model)
  
  cat("Full model", criterion, ":", round(full_criterion, 2), "\n")
  cat("Selected model", criterion, ":", round(selected_criterion, 2), "\n")
  cat("Selected features:", length(selected_features), "\n")
  cat("True positives:", true_positives, "/", length(data$true_features), "\n")
  cat("False positives:", false_positives, "\n")
  cat("Selected features:", paste(selected_features, collapse = ", "), "\n\n")
  
  return(list(
    model = step_model,
    selected_features = selected_features,
    criterion_value = selected_criterion,
    full_model = full_model
  ))
}

# Run both AIC and BIC selection
aic_results <- perform_ic_selection(X, y, "AIC")
bic_results <- perform_ic_selection(X, y, "BIC")

# ==============================================================================
# METHOD 4: MARS (MULTIVARIATE ADAPTIVE REGRESSION SPLINES)
# Uses piecewise linear functions and automatic feature selection
# ==============================================================================

perform_mars_selection <- function(X, y, max_terms = 21) {
  cat("=== MARS Feature Selection ===\n")
  
  # Create data frame
  df <- data.frame(y = y, X)
  
  # Fit MARS model with cross-validation
  mars_model <- earth(y ~ ., data = df, 
                      nk = max_terms,           # Maximum number of model terms
                      degree = 2,              # Allow interactions up to degree 2
                      pmethod = "cv",          # Use cross-validation for pruning
                      nfold = 10,              # 10-fold cross-validation
                      ncross = 3,              # Number of cross-validation repetitions
                      trace = 0)               # Suppress output
  
  # Extract selected features from the model terms
  model_terms <- mars_model$dirs
  feature_names <- colnames(X)
  
  # Identify which features appear in the final model
  selected_features <- character(0)
  if(!is.null(model_terms) && nrow(model_terms) > 1) {  # More than just intercept
    for(i in 2:nrow(model_terms)) {  # Skip intercept row
      term_features <- feature_names[model_terms[i, ] != 0]
      selected_features <- union(selected_features, term_features)
    }
  }
  
  # Calculate selection performance
  true_positives <- sum(selected_features %in% data$true_features)
  false_positives <- sum(selected_features %in% data$noise_features)
  
  # Model performance metrics
  rsq <- mars_model$rsq
  gcv_score <- mars_model$gcv
  
  cat("Number of terms in final model:", nrow(model_terms), "\n")
  cat("R-squared:", round(rsq, 3), "\n")
  cat("GCV score:", round(gcv_score, 3), "\n")
  cat("Selected features:", length(selected_features), "\n")
  cat("True positives:", true_positives, "/", length(data$true_features), "\n")
  cat("False positives:", false_positives, "\n")
  cat("Selected features:", paste(selected_features, collapse = ", "), "\n\n")
  
  return(list(
    model = mars_model,
    selected_features = selected_features,
    rsq = rsq,
    gcv = gcv_score
  ))
}

mars_results <- perform_mars_selection(X, y)

# ==============================================================================
# COMPARISON SUMMARY
# Compare all methods on their feature selection performance
# ==============================================================================

create_comparison_summary <- function() {
  cat("=== FEATURE SELECTION COMPARISON ===\n")
  
  methods <- list(
    "LASSO" = lasso_results$selected_features,
    "Tree-based" = tree_results$selected_features,
    "AIC" = aic_results$selected_features,
    "BIC" = bic_results$selected_features,
    "MARS" = mars_results$selected_features
  )
  
  # Calculate metrics for each method
  summary_table <- data.frame(
    Method = names(methods),
    Selected = sapply(methods, length),
    True_Positive = sapply(methods, function(x) sum(x %in% data$true_features)),
    False_Positive = sapply(methods, function(x) sum(x %in% data$noise_features)),
    Precision = sapply(methods, function(x) {
      if(length(x) == 0) return(0)
      sum(x %in% data$true_features) / length(x)
    }),
    Recall = sapply(methods, function(x) {
      sum(x %in% data$true_features) / length(data$true_features)
    }),
    stringsAsFactors = FALSE
  )
  
  # Calculate F1 score
  summary_table$F1_Score <- with(summary_table, {
    f1 <- 2 * (Precision * Recall) / (Precision + Recall)
    ifelse(is.na(f1), 0, f1)
  })
  
  print(summary_table)
  
  cat("\nTrue signal features:", paste(data$true_features, collapse = ", "), "\n")
  if(length(data$interaction_features) > 0) {
    cat("Features with interactions:", paste(data$interaction_features, collapse = ", "), "\n")
  }
  cat("Noise features:", paste(data$noise_features, collapse = ", "), "\n")
  
  return(summary_table)
}

comparison_results <- create_comparison_summary()

# ==============================================================================
# PRACTICAL USAGE NOTES
# ==============================================================================

cat("\n=== PRACTICAL USAGE GUIDELINES ===\n",
    "1. LASSO: Best for high-dimensional data, provides sparse solutions\n",
    "2. Tree-based: Good for non-linear relationships and interactions\n",
    "3. AIC/BIC: Classical approach, BIC tends to select fewer features\n",
    "4. MARS: Handles non-linearity and interactions automatically\n\n")

cat("Distribution-specific recommendations:\n",
    "- Normal data: All methods work well, LASSO often preferred\n",
    "- Count data (Poisson): Trees and MARS handle non-normality better\n",
    "- Time-to-event (Exponential): MARS adapts to skewed distributions\n",
    "- Mixed distributions: MARS most robust, Trees good alternative\n\n")

cat("To test different distributions, change distribution_type parameter:\n",
    'data <- generate_feature_selection_data(distribution_type = "poisson")\n',
    'Available types: "normal", "poisson", "binomial", "exponential", "gamma", "mixed"\n')
