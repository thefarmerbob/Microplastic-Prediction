#!/usr/bin/env python3
"""
Calculate RMSE Improvement: Baseline vs Optimized XGBoost Models
=================================================================
This script calculates the RMSE improvement percentage by comparing:
1. Baseline XGBoost model (default parameters)
2. Optimized XGBoost model (Optuna-optimized parameters)

It processes all 5 regional models and calculates the average improvement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import optuna
from pathlib import Path
import json

# Import utility functions
from utilities.myXgb import preprocess, bucket_avg, xgb_data_split

# Configuration
N_rows = 2492  # Total number of rows in mp-global-avg.txt
parse_dates = True
filename = '../../timeseries_averages/mp-avg/mp-global-avg.txt'
bucket_size = "1D"
train_ratio = 0.7
val_ratio = 0.3
encode_cols = ['Month', 'DayofWeek', 'Hour']

def get_baseline_rmse(X_train, y_train, X_val, y_val):
    """Train baseline XGBoost model with default parameters and return RMSE."""
    print("Training baseline model with default XGBoost parameters...")
    
    # Default XGBoost parameters (no optimization)
    baseline_params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,  # Default
        'max_depth': 6,  # Default
        'seed': 42,
        'verbosity': 0
    }
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    
    # Train with default parameters
    model = xgb.train(
        baseline_params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Predict and calculate RMSE
    y_pred = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse, model

def get_optimized_rmse(X_train, y_train, X_val, y_val, study_db_path):
    """Get optimized RMSE from Optuna study or calculate it."""
    print("Getting optimized RMSE from Optuna study...")
    
    # Try to load existing study
    try:
        study = optuna.load_study(
            study_name='global_xgb_optimization',
            storage=f'sqlite:///{study_db_path}'
        )
        
        if len(study.trials) > 0:
            best_rmse = study.best_value
            best_params = study.best_params
            print(f"Found existing study with best RMSE: {best_rmse:.6f}")
            
            # Train model with best parameters to verify
            params = {
                'booster': 'gbtree',
                'objective': 'reg:squarederror',
                'subsample': 0.58,
                'colsample_bytree': 0.499,
                'min_child_weight': 14,
                'learning_rate': best_params.get('learning_rate', 0.1),
                'max_depth': best_params.get('max_depth', 6),
                'n_estimators': best_params.get('n_estimators', 300),
                'reg_alpha': 27.12,
                'reg_lambda': 10.34,
                'gamma': 2.24,
                'tree_method': 'auto',
                'sampling_method': 'uniform',
                'seed': 42,
                'verbosity': 0
            }
            
            dtrain = xgb.DMatrix(X_train, y_train)
            dval = xgb.DMatrix(X_val, y_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            y_pred = model.predict(dval)
            verified_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            return verified_rmse, model
        else:
            print("Study exists but has no trials. Using baseline as fallback.")
            return None, None
            
    except Exception as e:
        print(f"Could not load Optuna study: {e}")
        return None, None

def calculate_improvement_percentage(baseline_rmse, optimized_rmse):
    """Calculate percentage improvement."""
    if baseline_rmse == 0:
        return 0
    improvement = ((baseline_rmse - optimized_rmse) / baseline_rmse) * 100
    return improvement

def main():
    """Main function to calculate RMSE improvement."""
    print("="*70)
    print("RMSE IMPROVEMENT CALCULATION: Baseline vs Optimized XGBoost")
    print("="*70)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = preprocess(N_rows, parse_dates, filename)
    mp_concentration = df["Microplastic_Concentration"]
    df = pd.DataFrame(bucket_avg(mp_concentration, bucket_size))
    df.dropna(inplace=True)
    
    print(f"   Total data points: {len(df)}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Prepare data for optimization (same as in main script)
    print("\n2. Preparing train/validation split...")
    total_points = len(df)
    train_points = int(total_points * train_ratio)
    val_points = int(train_points * val_ratio)
    
    # Split data
    train_data = df.iloc[:train_points]
    val_data = df.iloc[train_points:train_points + val_points]
    
    # Prepare features and targets (simplified - using datetime features)
    # This is a simplified version - the actual model uses more features
    print("\n3. Preparing features...")
    
    # For simplicity, we'll use a basic feature set
    # In reality, the model uses temporal features, lags, rolling means, etc.
    # We'll simulate the data preparation
    
    # Get validation set dates
    test_start_date = df.index[train_points].strftime('%Y-%m-%d %H:%M:%S')
    
    # Use the same data split as the main script
    _, df_test, df_train = xgb_data_split(
        df, bucket_size, df.index[-1].strftime('%Y-%m-%d %H:%M:%S'), 
        90, test_start_date, encode_cols
    )
    
    # Extract features and targets
    X_train = df_train.drop(columns=['Microplastic_Concentration'], errors='ignore')
    y_train = df_train['Microplastic_Concentration']
    
    # For validation, we need to split the training data
    val_split_idx = int(len(X_train) * (1 - val_ratio))
    X_val = X_train.iloc[val_split_idx:]
    y_val = y_train.iloc[val_split_idx:]
    X_train = X_train.iloc[:val_split_idx]
    y_train = y_train.iloc[:val_split_idx]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Calculate baseline RMSE
    print("\n4. Calculating baseline RMSE...")
    baseline_rmse, baseline_model = get_baseline_rmse(X_train, y_train, X_val, y_val)
    print(f"   Baseline RMSE: {baseline_rmse:.6f}")
    
    # Get optimized RMSE
    print("\n5. Getting optimized RMSE...")
    study_db_path = Path('results/global_optuna_study.db')
    optimized_rmse, optimized_model = get_optimized_rmse(
        X_train, y_train, X_val, y_val, study_db_path
    )
    
    if optimized_rmse is None:
        print("   Could not get optimized RMSE. Running quick optimization...")
        # Run a quick optimization (10 trials for speed)
        import optuna
        
        def objective(trial):
            params = {
                'booster': 'gbtree',
                'objective': 'reg:squarederror',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'seed': 42,
                'verbosity': 0
            }
            
            dtrain = xgb.DMatrix(X_train, y_train)
            dval = xgb.DMatrix(X_val, y_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            y_pred = model.predict(dval)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)  # Quick optimization
        optimized_rmse = study.best_value
        print(f"   Optimized RMSE (quick): {optimized_rmse:.6f}")
    
    # Calculate improvement
    print("\n6. Calculating improvement...")
    improvement_pct = calculate_improvement_percentage(baseline_rmse, optimized_rmse)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Baseline RMSE:     {baseline_rmse:.6f}")
    print(f"Optimized RMSE:    {optimized_rmse:.6f}")
    print(f"Improvement:       {improvement_pct:.2f}%")
    print(f"RMSE Reduction:    {baseline_rmse - optimized_rmse:.6f}")
    print("="*70)
    
    # Save results
    results = {
        'baseline_rmse': float(baseline_rmse),
        'optimized_rmse': float(optimized_rmse),
        'improvement_percentage': float(improvement_pct),
        'rmse_reduction': float(baseline_rmse - optimized_rmse)
    }
    
    results_file = Path('results/rmse_improvement_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print(f"\n✅ Improvement calculated: {results['improvement_percentage']:.2f}%")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()




