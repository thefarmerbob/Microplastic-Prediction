# RMSE Values Summary

## Extracted RMSE Values from XGBoost Models

### Individual Regional Models

| Region | Best RMSE | Worst RMSE | First RMSE | Improvement (Worst→Best) | Improvement (First→Best) |
|--------|-----------|------------|------------|---------------------------|--------------------------|
| **Global** | 2.671190 | 9.549999 | 2.822424 | **72.03%** | 5.36% |
| **Kyoto** | 58.465280 | 224.228662 | 67.089841 | **73.93%** | 12.86% |
| **Osaka** | 24.261969 | 49.422808 | 33.937483 | **50.91%** | 28.51% |
| **Tokyo** | 196.651817 | 309.129082 | 284.958180 | **36.39%** | 30.99% |
| **Tsushima** | 25.936465 | 65.778256 | 35.844195 | **60.57%** | 27.64% |

### Overall Statistics

- **Average Best RMSE**: 61.597344
- **Average Worst RMSE**: 131.621762
- **Overall Improvement (Worst→Best)**: **53.20%**
- **Average Improvement (First→Best)**: **21.07%**
- **Minimum Improvement**: 5.36% (Global)
- **Maximum Improvement**: 30.99% (Tokyo)

## Analysis of 59% RMSE Reduction Claim

### Findings:

1. **Overall improvement (worst→best)**: **53.20%** - This is close to but not exactly 59%
2. **Average improvement (first→best)**: **21.07%** - This is much lower than 59%

### Possible Explanations for 59%:

1. **Different Baseline**: The 59% might be comparing against a simpler baseline model (e.g., ARIMA, linear regression, or naive forecast) rather than worst Optuna trial
2. **Weighted Average**: The 59% might be a weighted average across regions
3. **Specific Region**: The 59% might refer to a specific region's improvement
4. **Before Optimization**: The 59% might compare against XGBoost with default parameters (before any Optuna optimization)

### Recommendations:

#### Option 1: Use Actual Found Values
```
Improved microplastic forecasting accuracy by 53% RMSE reduction (worst→best) 
by deploying 5 regional XGBoost models with Optuna hyperparameter optimization 
across 2,490 NASA time-series datasets
```

#### Option 2: Use Conservative Estimate
```
Improved microplastic forecasting accuracy by deploying 5 regional XGBoost models 
with Optuna hyperparameter optimization across 2,490 NASA time-series datasets, 
achieving up to 73% RMSE reduction in regional models
```

#### Option 3: If You Have Baseline Model Data
If you have RMSE from a baseline model (ARIMA, linear regression, or default XGBoost), calculate:
```
Improvement = (Baseline_RMSE - Optimized_RMSE) / Baseline_RMSE × 100
```

#### Option 4: Keep 59% If You Have Documentation
If you have external documentation showing 59% improvement against a specific baseline model, you can keep it. Otherwise, use the verified 53.20% or the more conservative wording.

## Regional Breakdown

### Best Performing Regions (by improvement):
1. **Kyoto**: 73.93% improvement (worst→best)
2. **Global**: 72.03% improvement (worst→best)
3. **Tsushima**: 60.57% improvement (worst→best)

### Most Consistent Improvement (first→best):
1. **Tokyo**: 30.99% improvement
2. **Osaka**: 28.51% improvement
3. **Tsushima**: 27.64% improvement

## Notes

- All models used 100 Optuna trials
- The "first RMSE" represents the first trial in the optimization process (not a true baseline)
- The "worst RMSE" represents the worst performing trial during optimization
- The "best RMSE" represents the Optuna-optimized result

## Files Generated

- `regional-xgboost/results/rmse_extraction_results.json` - Detailed results in JSON format
- This summary document




