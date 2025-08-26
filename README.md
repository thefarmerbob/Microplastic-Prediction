# MP-Prediction: Microplastic Prediction using CYGNSS Data

This project implements machine learning models to predict microplastic concentrations using CYGNSS (Cyclone Global Navigation Satellite System) data. The project includes advanced deep learning models, clustering analysis, and comprehensive time series visualization tools.

## Project Structure

```
MP-Prediction/
├── Time-Series-ARIMA-XGBOOST-RNN/
│   ├── dbscan_timeseries/        # DBSCAN clustering analysis and time series visualization
│   ├── regional-lstm/            # Advanced LSTM models with attention mechanisms
│   ├── regional-xgboost/         # Regional XGBoost models with Optuna optimization
│   ├── timeseries_averages/      # Time series data processing and regional averages
│   ├── util.py                   # Utility functions
│   ├── requirements.txt          # Python dependencies for the ML models
│   └── venv/                     # Virtual environment
├── requirements.txt              # Main project dependencies
└── README.md                     # This file
```

## Features

- **Advanced Deep Learning**: Regional LSTM models with attention mechanisms and ConvLSTM architectures
- **Regional XGBoost Models**: Separate optimized models for global, Kyoto, Osaka, Tokyo, and Tsushima regions
- **DBSCAN Clustering**: Time series clustering analysis with visualization
- **Comprehensive Visualization**: Time series plots, GIF animations, and spatial analysis
- **Optuna Optimization**: Hyperparameter optimization for XGBoost models
- **Attention Analysis**: Occlusion analysis and attention mechanism interpretation
- **Data Processing**: CYGNSS data processing, NetCDF conversion, and feature engineering

## Models and Components

### 1. Regional LSTM Models (`regional-lstm/`)
- **Purpose**: Advanced deep learning models for microplastic prediction
- **Key Features**:
  - ConvLSTM with attention mechanisms
  - Spatial-temporal analysis for Japan and Tsushima regions
  - Ground truth visualization utilities
  - Occlusion analysis for model interpretability
- **Trained Models**: `*.pth` files for different regions
- **Analysis Scripts**: Attention analysis, occlusion studies

### 2. Regional XGBoost Models (`regional-xgboost/`)
- **Purpose**: Region-specific XGBoost models with Optuna optimization
- **Regions**: Global, Kyoto, Osaka, Tokyo, Tsushima
- **Features**: 
  - Temporal features (lags, rolling means, trends)
  - Datetime features (month, day of week, hour)
  - Hyperparameter optimization with 100 trials per region
- **Outputs**: Feature importance plots, prediction visualizations, optimized parameters

### 3. DBSCAN Time Series Analysis (`dbscan_timeseries/`)
- **Purpose**: Clustering analysis of microplastic time series data
- **Features**:
  - DBSCAN clustering implementation
  - Time series visualization with GIF animations
  - Cropped time series analysis (last 365 days)
  - Spatial clustering visualization

### 4. Time Series Data Processing (`timeseries_averages/`)
- **Purpose**: Data preprocessing and regional average calculation
- **Components**:
  - Global microplastic averages (`mp-avg/`)
  - Regional averages for different geographical areas
  - Section cropping and data processing utilities

## Data

The project processes microplastic concentration data derived from CYGNSS satellite observations:
- **Time Series Averages**: Processed regional and global average data stored in `timeseries_averages/`
- **Global Data**: `mp-avg/mp-global-avg.txt`
- **Regional Data**: Separate files for Kyoto, Osaka, Tokyo, and Tsushima regions
- **Format**: Timestamped data with CYGNSS observation periods
- **Visualization**: Time series plots and spatial distribution maps

## Usage

### Regional XGBoost Models

Run individual regional models:
```bash
cd Time-Series-ARIMA-XGBOOST-RNN/regional-xgboost/
python model_scripts/Gpower_Xgb_Main_global.py   # Global model
python model_scripts/Gpower_Xgb_Main_kyoto.py    # Kyoto region
python model_scripts/Gpower_Xgb_Main_osaka.py    # Osaka region
python model_scripts/Gpower_Xgb_Main_tokyo.py    # Tokyo region
python model_scripts/Gpower_Xgb_Main_tsushima.py # Tsushima region
```

Run all regional models:
```bash
cd Time-Series-ARIMA-XGBOOST-RNN/regional-xgboost/
python utilities/run_all_regional_models.py
```

### Regional LSTM Models

Train ConvLSTM models:
```bash
cd Time-Series-ARIMA-XGBOOST-RNN/regional-lstm/
python training_scripts/sa_convlstm_microplastics.py     # Main ConvLSTM training
python training_scripts/sa_convlstm_microplastics_narrow.py  # Narrow region training
python training_scripts/sa_tsushima.py                   # Tsushima-specific model
```

Run analysis scripts:
```bash
cd Time-Series-ARIMA-XGBOOST-RNN/regional-lstm/
python attention_analysis/attention_analysis_final.py    # Attention mechanism analysis
python attention_analysis/tsushima_occlusion_analysis.py # Occlusion analysis
```

### DBSCAN Clustering Analysis

```bash
cd Time-Series-ARIMA-XGBOOST-RNN/dbscan_timeseries/
python dbscan.py                    # Main DBSCAN clustering
python dbscan-tsushima.py          # Tsushima-specific clustering
python create_efficient_gif.py     # Create time series animations
```

## Dependencies

The project includes comprehensive dependencies for machine learning, deep learning, and scientific computing:

### Core Data Science
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

### Machine Learning & Deep Learning
- tensorflow >= 2.8.0
- torch >= 1.12.0 (PyTorch)
- xgboost >= 1.5.0
- optuna >= 3.0.0
- statsmodels >= 0.12.0

### Visualization & Data Processing
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- imageio >= 2.9.0
- opencv-python >= 4.5.0

### Scientific Data Handling
- xarray >= 0.20.0
- netCDF4 >= 1.5.7
- h5netcdf >= 0.11.0
- h5py >= 3.0.0

Install dependencies:
```bash
pip install -r requirements.txt
cd Time-Series-ARIMA-XGBOOST-RNN/
pip install -r requirements.txt  # Additional ML-specific dependencies
```

## Results and Outputs

The project generates comprehensive analysis outputs:

### XGBoost Models
- **Parameter Files**: `*.csv` files with optimized hyperparameters for each region
- **Visualizations**: Feature importance plots and prediction vs. actual comparisons
- **Optuna Studies**: `*.db` files containing optimization histories

### LSTM Models
- **Trained Models**: `*.pth` files with trained ConvLSTM weights
- **Predictions**: `*.npy` arrays with test predictions and targets
- **Analysis Results**: Attention maps and occlusion analysis visualizations

### DBSCAN Analysis
- **Clustering Results**: Time series clustering visualizations
- **Animations**: GIF files showing temporal evolution of microplastic distributions
- **Spatial Analysis**: Cropped time series analysis for focused regions

### Visualization Outputs
- Time series plots showing microplastic concentration trends
- Regional comparison charts
- Spatial distribution maps
- Attention mechanism visualizations
- Model performance metrics and plots

## License

This project is for research purposes in microplastic prediction using satellite data.

## Contact

For questions or contributions, please contact the project maintainers. 