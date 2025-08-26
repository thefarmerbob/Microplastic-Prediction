# Regional and Global XGBoost Microplastic Prediction Models

This directory contains separate XGBoost prediction models for the global average and each regional microplastic concentration dataset. Each model is optimized specifically for its respective dataset using Optuna hyperparameter optimization.

## Directory Structure

```
regional-xgboost/
├── model_scripts/          # Main training scripts for each region
│   ├── Gpower_Xgb_Main_global.py
│   ├── Gpower_Xgb_Main_kyoto.py
│   ├── Gpower_Xgb_Main_osaka.py
│   ├── Gpower_Xgb_Main_tokyo.py
│   └── Gpower_Xgb_Main_tsushima.py
├── utilities/              # Utility functions and helper scripts
│   ├── myXgb.py           # Core XGBoost utility functions
│   ├── run_all_regional_models.py
│   ├── plot_all_regions.py
│   ├── debug_data_split.py
│   └── test_model_integrity.py
├── results/               # All output files (plots, parameters, databases)
│   ├── *.png             # Model plots and visualizations
│   ├── *.csv             # Optimized parameters
│   └── *.db              # Optuna study databases
└── README.md
```

## Files Overview

### Model Scripts (`model_scripts/`)

#### 1. `Gpower_Xgb_Main_global.py`
- **Purpose**: Predicts microplastic concentration for the Global average
- **Data Source**: `../../timeseries_averages/mp-avg/mp-global-avg.txt`
- **Features**: 
  - Temporal features (lags: 1, 7, 30 days)
  - Rolling means (7, 30 days)
  - Trend features (difference, percentage change)
  - Datetime features (Month, DayofWeek, Hour)
- **Outputs**:
  - `results/Global_Optuna_Optimized_All_Features.png` - Feature importance plot
  - `results/Global_Optuna_Optimized_Model.png` - Prediction vs actual plot
  - `results/global_optuna_optimized_parameters.csv` - Optimized parameters
  - `results/global_optuna_study.db` - Optuna study database

#### 2. `Gpower_Xgb_Main_kyoto.py`
- **Purpose**: Predicts microplastic concentration for the Kyoto region
- **Data Source**: `../../timeseries_averages/regional-averages/mp-kyoto-avg.txt`
- **Features**: Same as Global model
- **Outputs**:
  - `results/Kyoto_Optuna_Optimized_All_Features.png` - Feature importance plot
  - `results/Kyoto_Optuna_Optimized_Model.png` - Prediction vs actual plot
  - `results/kyoto_optuna_optimized_parameters.csv` - Optimized parameters
  - `results/kyoto_optuna_study.db` - Optuna study database

#### 3. `Gpower_Xgb_Main_osaka.py`
- **Purpose**: Predicts microplastic concentration for the Osaka region
- **Data Source**: `../../timeseries_averages/regional-averages/mp-osaka-avg.txt`
- **Features**: Same as Global model
- **Outputs**:
  - `results/Osaka_Optuna_Optimized_All_Features.png` - Feature importance plot
  - `results/Osaka_Optuna_Optimized_Model.png` - Prediction vs actual plot
  - `results/osaka_optuna_optimized_parameters.csv` - Optimized parameters
  - `results/osaka_optuna_study.db` - Optuna study database

#### 4. `Gpower_Xgb_Main_tokyo.py`
- **Purpose**: Predicts microplastic concentration for the Tokyo region
- **Data Source**: `../../timeseries_averages/regional-averages/mp-tokyo-avg.txt`
- **Features**: Same as Global model
- **Outputs**:
  - `results/Tokyo_Optuna_Optimized_All_Features.png` - Feature importance plot
  - `results/Tokyo_Optuna_Optimized_Model.png` - Prediction vs actual plot
  - `results/tokyo_optuna_optimized_parameters.csv` - Optimized parameters
  - `results/tokyo_optuna_study.db` - Optuna study database

#### 5. `Gpower_Xgb_Main_tsushima.py`
- **Purpose**: Predicts microplastic concentration for the Tsushima region
- **Data Source**: `../../timeseries_averages/regional-averages/mp-tsushima-avg.txt`
- **Features**: Same as Global model
- **Outputs**:
  - `results/Tsushima_Optuna_Optimized_All_Features.png` - Feature importance plot
  - `results/Tsushima_Optuna_Optimized_Model.png` - Prediction vs actual plot
  - `results/tsushima_optuna_optimized_parameters.csv` - Optimized parameters
  - `results/tsushima_optuna_study.db` - Optuna study database

### Utility Scripts (`utilities/`)

#### `myXgb.py`
- **Purpose**: Core XGBoost utility functions
- **Functions**: `xgb_data_split`, `xgb_importance`, `xgb_forecasts_plot`, `xgb_forecasts_plot_with_actual`

#### `run_all_regional_models.py`
- **Purpose**: Batch processing script to run all regional models sequentially
- **Usage**: Executes all five models in order and provides a summary of results

#### `plot_all_regions.py`
- **Purpose**: Creates combined visualization of all regional data
- **Output**: `results/All_Regions_Combined_Plot.png`

#### `debug_data_split.py`
- **Purpose**: Debugging script for data splitting issues
- **Usage**: Test data preprocessing and splitting logic

#### `test_model_integrity.py`
- **Purpose**: Validates model training and prediction integrity
- **Usage**: Test model performance and data handling

## Data Format

The regional data files have a different format than the global average:
- **Format**: `YYYYMMDD-HHMMSS-eYYYYMMDD-HHMMSS: value`
- **Example**: `20180816-120000-e20180816-120000: 12780.02`
- **Handling**: NaN values are automatically skipped during preprocessing

## Model Configuration

All models use the same optimized configuration:
- **Train/Test Split**: 70% training, 30% testing
- **Validation Split**: 30% of training data for validation
- **Optimization**: Optuna with 100 trials
- **Parameters Optimized**: learning_rate, max_depth, n_estimators
- **Fixed Parameters**: Based on best known values from global model

## Usage

### Individual Models

To run any of the models individually:

```bash
cd Time-Series-ARIMA-XGBOOST-RNN/regional-xgboost
python model_scripts/Gpower_Xgb_Main_global.py   # For Global
python model_scripts/Gpower_Xgb_Main_kyoto.py    # For Kyoto
python model_scripts/Gpower_Xgb_Main_osaka.py    # For Osaka
python model_scripts/Gpower_Xgb_Main_tokyo.py    # For Tokyo
python model_scripts/Gpower_Xgb_Main_tsushima.py # For Tsushima
```

### Batch Processing

To run all regional models sequentially:

```bash
cd Time-Series-ARIMA-XGBOOST-RNN/regional-xgboost
python utilities/run_all_regional_models.py
```

This will execute all five models in order and provide a summary of results.

### Utility Scripts

```bash
# Create combined regional plot
python utilities/plot_all_regions.py

# Debug data splitting
python utilities/debug_data_split.py

# Test model integrity
python utilities/test_model_integrity.py
```

## Key Features

1. **Custom Preprocessing**: Each model includes a `preprocess_regional_data()` function that handles the specific regional data format
2. **Smart Optuna Optimization**: Each region gets its own optimized hyperparameters with intelligent caching:
   - If an Optuna study already exists with completed trials, it uses the best parameters from the existing study
   - If no study exists, it runs a new optimization with 100 trials
3. **Feature Engineering**: Temporal features, rolling statistics, and trend analysis
4. **Performance Metrics**: RMSE for both validation and test sets
5. **Visualization**: Prediction plots and feature importance analysis

## Dependencies

- numpy
- pandas
- xgboost
- optuna
- matplotlib
- scikit-learn
- scipy

## Notes

- Each model creates its own Optuna study database to avoid conflicts
- The preprocessing function handles the regional data format automatically
- All models use the same feature engineering approach for consistency
- Results are saved with region-specific naming to avoid conflicts
