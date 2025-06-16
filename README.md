# Modeling Framework Repository

## Tidymodels Framework 

Production-ready machine learning pipeline for R with automated preprocessing, model training, and ensemble methods.

### Features

- Automatic problem detection (regression/classification)
- Parallel processing with 8+ algorithms
- Ensemble stacking and model interpretability
- Advanced preprocessing and hyperparameter tuning

### Quick Start

#### Install Dependencies
```r
install.packages(c("tidymodels", "stacks", "workflowsets", "rules", 
                   "baguette", "finetune", "textrecipes", "themis", 
                   "DALEXtra", "vip", "parallel", "doParallel", "earth"))
```

#### Basic Setup
1. Replace demonstration data:
```r
data <- read_csv("your_data.csv")
```

2. Configure workflow:
```r
OUTCOME_VAR <- "your_target_variable"
OUTCOME_TYPE <- "auto"  # auto, regression, binary, multiclass
```

3. Run all code sections in order.

### Configuration

```r
# Essential settings
OUTCOME_VAR <- "Sale_Price"           # Target variable
TEST_SPLIT <- 0.2                     # Test set proportion
CV_FOLDS <- 8                         # Cross-validation folds

# Optional features
FEATURE_ENGINEERING <- FALSE          # Advanced preprocessing
USE_RACING <- TRUE                    # Efficient tuning
STACK_ENSEMBLE <- TRUE               # Ensemble modeling
PARAMETER_COMBINATIONS <- 10         # Tuning iterations
```

### Output

- **Models**: Linear, Random Forest, XGBoost, SVM, Neural Networks, etc.
- **Metrics**: RMSE/R² (regression), ROC AUC/Accuracy (classification)
- **Objects**: `grid_results`, `final_fit`, `fitted_stack`, `test_metrics`

### Troubleshooting

- **Memory issues**: Reduce `PARAMETER_COMBINATIONS`
- **Long runtime**: Enable `USE_RACING`, reduce `CV_FOLDS`
- **Model failures**: Check outcome variable format

Built on [tidymodels](https://www.tidymodels.org/)
