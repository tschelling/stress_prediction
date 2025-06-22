import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns # Make sure seaborn is imported if used in plot_features_timeseries_flat
from typing import List, Dict, Any # For type hinting
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
import joblib
import os
# import math # No longer used directly in this script for plotting

# Added for Neural Network
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1_l2
from sklearn.ensemble import RandomForestRegressor # Added for RandomForest
from keras.callbacks import EarlyStopping, TensorBoard # Import TensorBoard
from scikeras.wrappers import KerasRegressor

import regression_data_preparer
import importlib
importlib.reload(regression_data_preparer)  # Ensure the latest version of PanelDataPreparer is used
from regression_data_preparer import RegressionDataPreparer
from sklearn.base import clone # Added for VotingRegressor

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


#--------------------------------------------------------------------------------------------------------------------
#region Configure 
#--------------------------------------------------------------------------------------------------------------------

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow INFO and WARNING messages for cleaner output


# --- Data & Model Configuration ---

TARGET_VARIABLES = {'interest_income_to_assets':'bank', 'interest_expense_to_assets':'bank',
                   'non_interest_income_to_assets':'bank', 'non_interest_expense_to_assets':'bank',
                   'net_charge_offs_to_loans_and_leases':'bank'}
FEATURE_VARIABLES = {'deposit_ratio':'bank', 'loan_to_asset_ratio':'bank', 'log_total_assets':'bank', 
                     'cpi_qoq':'macro',      'gdp_qoq':'macro',     'unemployment':'macro', 'household_delinq':'macro', 
                     'tbill_3m':'macro',     'tbill_10y':'macro', 'sp500_qoq':'macro',
                     'corp_bond_spread':'macro', 'vix_qoq':'macro', 'is_structural_break':'bank',
                     # 'dep_small_3m_less_to_assets':'bank',
                     # 'dep_small_3m_1y_to_assets':'bank',
                     # 'dep_small_1y_3y_to_assets':'bank',
                     # 'dep_small_3y_more_to_assets':'bank',
                     # 'dep_large_3m_less_to_assets':'bank',
                     # 'dep_large_3m_1y_to_assets':'bank',
                     # 'dep_large_1y_3y_to_assets':'bank',
                     # 'dep_large_3y_more_to_assets':'bank', 
                     # 'closed_end_first_liens_1_4_res_prop_3m_less_to_assets':'bank',
                     # 'closed_end_first_liens_1_4_res_prop_3m_1y_to_assets':'bank',
                     # 'closed_end_first_liens_1_4_res_prop_1y_3y_to_assets':'bank',
                     # 'closed_end_first_liens_1_4_res_prop_3y_5y_to_assets':'bank',
                     # 'closed_end_first_liens_1_4_res_prop_5y_15y_to_assets':'bank',
                     # 'closed_end_first_liens_1_4_res_prop_15y_more_to_assets':'bank',
                     # 'all_other_loans_3m_less_to_assets':'bank',
                     # 'all_other_loans_3m_1y_to_assets':'bank',
                     # 'all_other_loans_1y_3y_to_assets':'bank',
                     # 'all_other_loans_3y_5y_to_assets':'bank',
                     # 'all_other_loans_5y_15y_to_assets':'bank',
                     # 'all_other_loans_15y_more_to_assets':'bank'
                     } # Added more macro vars
FORECAST_HORIZONS = list(range(1, 2))
# Get feature variables as a list


c = {
    'TARGET_VARIABLES': TARGET_VARIABLES,     
    'FEATURE_VARIABLES': FEATURE_VARIABLES,             
    'INCLUDE_TIME_FE': True,                           
    'INCLUDE_BANK_FE': False,    
    'INCLUDE_STRUCTURAL_BREAK_DUMMY': True,               
    'OUTLIER_THRESHOLD_TARGET': 3.0,                    
    'MIN_OBS_PER_BANK': 12,             
    'CORRECT_STRUCTURAL_BREAKS_TOTAL_ASSETS': True,                
    'DATA_BEGIN': '2017-01-01',                              
    'DATA_END': None,                                  
    'RESTRICT_TO_NUMBER_OF_BANKS': 100,                 
    'RESTRICT_TO_BANK_SIZE': None,                      
    'RESTRICT_TO_MINIMAL_DEPOSIT_RATIO': None,          
    'RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO': None,     
    'INCLUDE_AUTOREGRESSIVE_LAGS': True,                
    'NUMBER_OF_LAGS_TO_INCLUDE': 8,                     
    'TRAIN_TEST_SPLIT_DIMENSION': 'date',               
    'TEST_SPLIT': "2022-04-01"                                  # Takes a number or a date in the format 'YYYY-MM-DD'
}



# --- Model Selection ---
# Options: None (all defined models), or a list of model names, e.g., ["XGBoost", "Ridge", "NeuralNetwork"]
# Available models: "XGBoost", "RandomForest", "DecisionTree", "Lasso", "Ridge", "ElasticNet", 
#                   "LinearRegression", "NeuralNetwork", "DummyRegressor", "RFE_Pipeline_RF", "LightGBM", "CatBoost"
MODELS_TO_RUN = ["LinearRegression", "XGBoost", "RandomForest",
                 "RFE_Pipeline_RF", "RFE_Pipeline_LR"] #"RFE_DTR_Pipeline_RF", "RFE_DTR_Pipeline_LR"] 



# --- Cross-validation & Hyperparameter Tuning ---
N_SPLITS_CV = 3
USE_RANDOM_SEARCH_CV = True 
N_ITER_RANDOM_SEARCH = 20

# --- Ensemble Configuration ---
POTENTIAL_BASE_MODELS_FOR_ENSEMBLE = ["XGBoost", "RandomForest", "DecisionTree", "LightGBM", "CatBoost"]

# --- TensorFlow/Keras Specifics ---
if tf.config.list_physical_devices('GPU'):
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print("TensorFlow Mixed Precision policy set to 'mixed_float16'.")

# --- Artifact Storage & Display ---
SAVE_ARTIFACTS = True
ARTIFACTS_BASE_DIR = "models_and_results1"
PLOT_RESULT_CHARTS = False
PRINT_FINAL_SUMMARY = True


# Helper function to create Neural Network model for KerasRegressor
def create_nn_model(**kwargs): # Add use_batch_norm
    """
    Creates a Keras Sequential model for regression.
    X_shape_ is automatically passed by KerasRegressor as a keyword argument and contains the input shape.
    Hyperparameters from the grid (or defaults) are passed as keyword arguments in kwargs.
    """
    # Meta parameters like X_shape_ and n_features_in_ are passed by scikeras
    # inside a 'meta' dictionary when the build function accepts **kwargs.
    meta = kwargs.pop('meta', {})  # Extract 'meta' dictionary, default to empty if not found
    X_shape_ = meta.get('X_shape_', None)
    n_features_in_ = meta.get('n_features_in_', None)
    # Remaining kwargs are the model hyperparameters (e.g., hidden_layers, neurons_per_layer)

    # Extract hyperparameters from kwargs, providing defaults
    input_dim = None
    if X_shape_ is not None and isinstance(X_shape_, tuple) and len(X_shape_) > 1:
        input_dim = X_shape_[1]
    elif n_features_in_ is not None:
        print(f"Warning: X_shape_ is invalid or not fully specified ({X_shape_}). Falling back to n_features_in_ ({n_features_in_}) for input_dim.")
        input_dim = n_features_in_
    
    if input_dim is None:
        raise ValueError(
            f"create_nn_model: Could not determine input_dim for the first Dense layer. "
            f"X_shape_ received: {X_shape_}, n_features_in_ received: {n_features_in_}. "
            "Ensure input data (X_train) is valid and non-empty."
        )

    hidden_layers = kwargs.get('hidden_layers', 1)
    neurons_per_layer = kwargs.get('neurons_per_layer', 64)
    activation = kwargs.get('activation', 'relu')
    optimizer_name = kwargs.get('optimizer', 'adam') # Renamed to avoid conflict with optimizer object
    learning_rate = kwargs.get('learning_rate', 0.001)
    dropout_rate = kwargs.get('dropout_rate', 0.0)
    l1_reg = kwargs.get('l1_reg', 0.0)
    l2_reg = kwargs.get('l2_reg', 0.0)
    use_batch_norm = kwargs.get('use_batch_norm', False)
    # Add other potential hyperparameters here as needed

    model = Sequential()
    # Use keras.Input as the first layer for explicit input shape definition
    model.add(keras.Input(shape=(input_dim,), name="input_layer"))
    model.add(Dense(neurons_per_layer, activation=activation,
                    kernel_regularizer=l1_l2(l1=float(l1_reg), l2=float(l2_reg)))) # Ensure float type
    if use_batch_norm:
        model.add(keras.layers.BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    for _ in range(hidden_layers - 1): # -1 because the first hidden layer is already added
        model.add(Dense(neurons_per_layer, activation=activation,
                        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
        if use_batch_norm:
            model.add(keras.layers.BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1, dtype='float32')) # Output layer for regression, ensure float32 output for stability

    if optimizer_name == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else: # Default or unsupported optimizer name
        opt = Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

def get_models_and_param_grids(use_random_search=False, n_iter_random_search=10):
    """Defines models and their hyperparameter grids for GridSearchCV."""
    # (Keep the existing get_models_and_param_grids function as it is in your original file)
    # For brevity, I'm not repeating the full function here, but it should remain unchanged
    # from your version that includes sklearn imports, XGBoost, Lasso, etc.
    # Ensure all necessary model classes are imported at the top of this function or globally.
    import sklearn.dummy # Example import, ensure all are present
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor # Ensure RandomForest is imported here too
    from sklearn.feature_selection import RFE # Import RFE
    from sklearn.pipeline import Pipeline # Import Pipeline

    from scipy.stats import uniform, randint

    models = {
        "DummyRegressor": sklearn.dummy.DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(random_state=42, max_iter=15000),
        "Ridge": Ridge(random_state=42, max_iter=15000),
        "ElasticNet": ElasticNet(random_state=42, max_iter=15000),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1, tree_method='hist'), # Removed enable_categorical
        "RandomForest": RandomForestRegressor( # Added RandomForest
            random_state=42, 
            n_jobs=-1 # Added RandomForest
            ),
        "FixedEffectsLR": LinearRegression(),
        "NeuralNetwork": KerasRegressor(
            model=create_nn_model,
            verbose=0,
            # Explicitly set default model parameters here.
            # These are used if tuning is skipped or fails.
            model__hidden_layers=1,
            model__neurons_per_layer=64, # Default, will be overridden by grid if tuning
            model__activation='relu',    # Default, will be overridden by grid if tuning
            model__optimizer='adam',
            model__learning_rate=0.001,
            model__dropout_rate=0.0,
            model__use_batch_norm=False, # Default for batch norm
            validation_split=0.1, # Default validation split for Keras internal validation
            # Early stopping callback for NN. scikeras handles validation split for callbacks during CV.
            # Monitor 'loss' if no explicit val split, 'val_loss' if scikeras provides one (e.g. during CV).
            # Let's use 'val_loss' as CV is typical, and 'loss' if CV is skipped.
            callbacks=[EarlyStopping(monitor='val_loss', patience=15, verbose=0, restore_best_weights=True, mode='min')] # Increased patience
        ),
        "RFE_Pipeline_RF": Pipeline([
            ('rfe', RFE(estimator=LinearRegression(), n_features_to_select=None)), 
            ('RandomForest', RandomForestRegressor(random_state=42, n_jobs=-1))
        ]),
        "RFE_Pipeline_LR": Pipeline([ # Added RFE with Linear Regression
            ('rfe', RFE(estimator=LinearRegression(), n_features_to_select=None)), 
            ('LinearRegression', LinearRegression()) # Final estimator
        ]),
        "RFE_DTR_Pipeline_RF": Pipeline([ # New: RFE with DecisionTreeRegressor estimator, final RF
            ('rfe', RFE(estimator=DecisionTreeRegressor(max_depth=5, random_state=42), n_features_to_select=None)),
            ('RandomForest', RandomForestRegressor(random_state=42, n_jobs=-1))
        ]),
        "RFE_DTR_Pipeline_LR": Pipeline([ # New: RFE with DecisionTreeRegressor estimator, final LR
            ('rfe', RFE(estimator=DecisionTreeRegressor(max_depth=5, random_state=42), n_features_to_select=None)),
            ('LinearRegression', LinearRegression())
        ]),
        "LightGBM": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False),
    }

    # Estimate max number of features for RFE tuning range
    # This is an approximation as actual features depend on FE/lags from DataPreparer
    # Consider time FE (e.g., 3 for quarter if OHE) and AR lags. Bank FE can be many, so RFE might not be ideal with it.
    approx_max_features = len(FEATURE_VARIABLES) + c.get('NUMBER_OF_LAGS_TO_INCLUDE', 0) + 3 
    if approx_max_features < 1: approx_max_features = 1 # Ensure at least 1

    if use_random_search:
        param_grids = {
            "Lasso": {'alpha': uniform(0.0001, 10 - 0.0001)},
            "Ridge": {'alpha': uniform(0.001, 100 - 0.001)},
            "ElasticNet": {
                'alpha': uniform(0.001, 10 - 0.001),
                'l1_ratio': uniform(0.1, 0.9 - 0.1)
            },
            "DecisionTree": {
                'max_depth': [None] + list(range(3, 21)), # Or use randint(3,21) if you want it to sample from this range
                'min_samples_split': randint(2, 21),
                'min_samples_leaf': randint(1, 11)
            },
            "XGBoost": {
                'n_estimators': randint(10, 151),
                'learning_rate': uniform(0.01, 0.3 - 0.01),
                'max_depth': randint(3, 12),
                'subsample': uniform(0.2, 1.0 - 0.2),
                'colsample_bytree': uniform(0.2, 1.0 - 0.2),
            },
            "RandomForest": {
                'n_estimators': randint(10, 201), # e.g., 10 to 200 trees
                'max_depth': [None] + list(range(5, 31)), # Or randint(5, 31)
                'min_samples_split': randint(2, 21),
                'min_samples_leaf': randint(1, 11)
            },
            "NeuralNetwork": {
                'model__hidden_layers': randint(1, 5), # 1 to 4 hidden layers
                'model__neurons_per_layer': randint(32, 257), # Neurons from 32 to 256
                'model__activation': ['relu', 'tanh'],
                'model__optimizer': ['adam', 'rmsprop'],
                'model__learning_rate': uniform(0.00005, 0.005), # Lower and wider range
                'model__dropout_rate': uniform(0.0, 0.6), # Dropout from 0.0 to 0.6
                'model__l1_reg': uniform(0.0, 0.05), # Wider range for L1
                'model__l2_reg': uniform(0.0, 0.01 - 0.0),
                'batch_size': [16, 32, 64, 128],
                'epochs': [50, 100, 150, 200], # Increased epochs
                'model__use_batch_norm': [True, False], # Tune BatchNormalization
                'validation_split': uniform(0.05, 0.2) # Tunable validation_split (0.05 to 0.25)
            },
            "RFE_Pipeline_RF": {
                'rfe__n_features_to_select': randint(max(1, approx_max_features // 4), max(2, approx_max_features + 1)), # Tune number of features from 1/4 to full
                # RandomForest params need to be prefixed with 'RandomForest__' (the name in the pipeline)
                'RandomForest__n_estimators': randint(10, 151),
                'RandomForest__max_depth': [None] + list(range(5, 21)),
                'RandomForest__min_samples_split': randint(2, 16),
                'RandomForest__min_samples_leaf': randint(1, 11)
            },
            "RFE_DTR_Pipeline_RF": { # Param grid for new RFE_DTR_Pipeline_RF
                'rfe__n_features_to_select': randint(max(1, approx_max_features // 4), max(2, approx_max_features + 1)),
                'RandomForest__n_estimators': randint(10, 151),
                'RandomForest__max_depth': [None] + list(range(5, 21)),
                'RandomForest__min_samples_split': randint(2, 16),
                'RandomForest__min_samples_leaf': randint(1, 11)
            },
            "RFE_Pipeline_LR": { # Param grid for RFE_Pipeline_LR
                'rfe__n_features_to_select': randint(max(1, approx_max_features // 4), max(2, approx_max_features + 1)),
                # LinearRegression itself has few hyperparameters to tune that drastically change stability beyond feature selection
                # 'LinearRegression__fit_intercept': [True, False] # Example if you wanted to tune LR params
            },
            "RFE_DTR_Pipeline_LR": { # Param grid for new RFE_DTR_Pipeline_LR
                'rfe__n_features_to_select': randint(max(1, approx_max_features // 4), max(2, approx_max_features + 1)),
                # 'LinearRegression__fit_intercept': [True, False]
            },
            "LightGBM": {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.2), # 0.01 to 0.21
                'num_leaves': randint(20, 100),
                'max_depth': [-1] + list(range(3, 15)),
                'subsample': uniform(0.6, 0.4), # 0.6 to 1.0
                'colsample_bytree': uniform(0.6, 0.4), # 0.6 to 1.0
            },
            "CatBoost": {
                'iterations': randint(50, 300),
                'learning_rate': uniform(0.01, 0.2),
                'depth': randint(3, 10),
                'l2_leaf_reg': uniform(1, 10), # 1 to 11
                'border_count': [32, 64, 128], # For numerical features
                'subsample': uniform(0.6, 0.4), # if bootstrap_type is 'Bernoulli' or 'Poisson'
            }
        }
    else: 
        param_grids = {
            "Lasso": {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 20]},
            "Ridge": {'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100]},
            "ElasticNet": {'alpha': [0.001, 0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
            "DecisionTree": {'max_depth': [None, 5, 10, 15, 20], 'min_samples_split': [2, 5, 10, 20],
                             'min_samples_leaf': [1, 5, 10]},
            "XGBoost": {
                'n_estimators': [20, 30, 35, 40, 45, 50, 100],
                'learning_rate': [0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9, 11],
            },
            "RandomForest": { # Example for GridSearchCV
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "NeuralNetwork": { # Example for GridSearchCV, usually too slow for NNs
                'model__hidden_layers': [1, 2],
                'model__neurons_per_layer': [32, 64],
                'model__activation': ['relu'],
                'model__optimizer': ['adam', 'rmsprop'],
                'model__learning_rate': [0.0005, 0.001, 0.005],
                'model__dropout_rate': [0.0, 0.2],
                'model__l2_reg': [0.0, 0.001, 0.005],
                'batch_size': [32, 64],
                'epochs': [70, 100],
                'model__use_batch_norm': [True, False],
                'validation_split': [0.1, 0.2]
            },
            "RFE_Pipeline_RF": {
                'rfe__n_features_to_select': [max(1, approx_max_features // 3), max(1, approx_max_features * 2 // 3), approx_max_features],
                'RandomForest__n_estimators': [50, 100],
                'RandomForest__max_depth': [None, 10],
                'RandomForest__min_samples_split': [2, 10],
                'RandomForest__min_samples_leaf': [1, 5]
            },
            "RFE_DTR_Pipeline_RF": { # Param grid for new RFE_DTR_Pipeline_RF (GridSearch)
                'rfe__n_features_to_select': [max(1, approx_max_features // 3), max(1, approx_max_features * 2 // 3), approx_max_features],
                'RandomForest__n_estimators': [50, 100],
                'RandomForest__max_depth': [None, 10],
                'RandomForest__min_samples_split': [2, 10],
                'RandomForest__min_samples_leaf': [1, 5]
            },
            "RFE_Pipeline_LR": { # Param grid for RFE_Pipeline_LR
                'rfe__n_features_to_select': [max(1, approx_max_features // 3), max(1, approx_max_features * 2 // 3), approx_max_features],
                # 'LinearRegression__fit_intercept': [True, False]
            },
            "RFE_DTR_Pipeline_LR": { # Param grid for new RFE_DTR_Pipeline_LR (GridSearch)
                'rfe__n_features_to_select': [max(1, approx_max_features // 3), max(1, approx_max_features * 2 // 3), approx_max_features],
                # 'LinearRegression__fit_intercept': [True, False]
            },
            "LightGBM": {
                'n_estimators': [50, 150],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 60],
                'max_depth': [-1, 10],
            },
            "CatBoost": {
                'iterations': [100, 200],
                'learning_rate': [0.03, 0.1],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [3, 5],
                'border_count': [64],
            }
        }
    return models, param_grids

#endregion

#--------------------------------------------------------------------------------------------------------------------
#region Model Training Helpers
#--------------------------------------------------------------------------------------------------------------------

def _tune_hyperparameters(model_name_key: str, model_instance, X_train: pd.DataFrame, y_train: pd.Series,
                          param_grid: dict, cv_splitter,
                          use_random_search: bool, n_iter_random_search: int):
    
    """
    Tunes hyperparameters for a given model using GridSearchCV or RandomizedSearchCV.
    Returns the best_estimator_ if tuning is successful, else None.
    """
    min_samples_for_cv = cv_splitter.get_n_splits() * 2 # A common heuristic, ensure enough samples per split
    if len(X_train) < min_samples_for_cv:
        print(f"    Skipping tuning for {model_name_key}: not enough samples ({len(X_train)}) for {cv_splitter.get_n_splits()} CV splits.")
        return None
    # Some models might fail or behave unexpectedly with constant target.
    # KerasRegressor might also struggle if loss doesn't change.
    # This check is a safeguard.
    if len(np.unique(y_train)) == 1 and model_name_key not in ["DummyRegressor", "DecisionTree", "XGBoost"]:
        print(f"    Skipping tuning for {model_name_key}: target variable is constant.")
        return None

    try:
        if use_random_search:
            search = RandomizedSearchCV(model_instance, param_distributions=param_grid,
                                        n_iter=n_iter_random_search, cv=cv_splitter,
                                        scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, error_score='raise',
                                        # Pass fit params for Keras if needed, e.g., validation_split for callbacks if not using CV's split
                                        # fit_params={'callbacks': [EarlyStopping(monitor='val_loss', patience=5)]} if 'NeuralNetwork' in model_name_key else None
                                        )
        else:
            search = GridSearchCV(model_instance, param_grid, cv=cv_splitter,
                                  scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise',
                                  # fit_params={'callbacks': [EarlyStopping(monitor='val_loss', patience=5)]} if 'NeuralNetwork' in model_name_key else None
                                 )
        # For KerasRegressor, callbacks are part of the model instance if set during KerasRegressor init.
        # scikeras handles the validation data for callbacks when used with scikit-learn CV.
        # So, explicit fit_params for callbacks here might be redundant if already in KerasRegressor.
        search.fit(X_train, y_train) # XGBoost early stopping during CV needs fit_params here if desired
        print(f"    Best params for {model_name_key}: {search.best_params_}")
        return search.best_estimator_
    except Exception as e:
        print(f"    Error during hyperparameter search for {model_name_key}: {e}.")
        return None

def _fit_model_with_tuning(model_name_key: str, model_instance, X_train: pd.DataFrame, y_train: pd.Series,
                           param_grid: dict | None, cv_splitter, 
                           use_random_search: bool, n_iter_random_search: int):
    """
    Handles the fitting process, including optional hyperparameter tuning.
    Returns the fitted model object or None if fitting fails.
    """
    if X_train.empty or (hasattr(X_train, 'shape') and X_train.shape[1] == 0):
        print(f"    Skipping fitting for {model_name_key}: X_train is empty or has no features.")
        return None

    fitted_model = None
    is_xgb = isinstance(model_instance, XGBRegressor)
    is_keras = isinstance(model_instance, KerasRegressor)
    fit_params = {}

    # Prepare fit_params for XGBoost early stopping if tuning is NOT used
    # If tuning is used, early stopping needs to be handled within the tuning function's fit call
    # For simplicity here, we'll only add eval_set if fitting the base model directly.
    # A more advanced approach would pass eval_set to RandomizedSearchCV/GridSearchCV's fit.
    # Given the current _tune_hyperparameters doesn't support eval_set in its fit,
    # we apply it only when fitting the base model instance.
    if is_xgb and hasattr(model_instance, 'early_stopping_rounds') and model_instance.early_stopping_rounds is not None:
         # Note: eval_set for tuning should ideally be a validation set, not the test set.
         # Using test set here for early stopping during final fit is common but can leak information.
         # For proper tuning with early stopping, a separate validation split within CV is needed.
         # Sticking to the current pattern for now, but noting this potential improvement.
         # We don't have X_test/y_test here, so we cannot add eval_set in this function.
         # This logic needs to be in train_evaluate_model or passed in.
         pass # Will handle eval_set in train_evaluate_model before calling this.

    if param_grid and cv_splitter:
        print(f"    Tuning {model_name_key}...")
        # For KerasRegressor, early stopping callbacks are already part of the model instance.
        # scikeras will use the CV splits to provide validation data to these callbacks.
        tuned_and_fitted_model = _tune_hyperparameters(
            model_name_key, model_instance, X_train, y_train, param_grid,
            cv_splitter, use_random_search, n_iter_random_search
        )
        if tuned_and_fitted_model is not None:
            fitted_model = tuned_and_fitted_model
        elif is_keras: # If tuning failed for Keras, still try to fit the base model
            print(f"    Fitting {model_name_key} with default parameters (tuning failed/skipped).")
            model_instance.fit(X_train, y_train) # KerasRegressor handles its own callbacks
            fitted_model = model_instance
        else:
            # Tuning was skipped or failed. Fit the original model_instance.
            print(f"    Fitting {model_name_key} with default parameters (tuning skipped or failed).")
            # Fit without eval_set here, as X_test/y_test are not available
            model_instance.fit(X_train, y_train)
            fitted_model = model_instance
    else:
        # No tuning. Fit the original model_instance.
        print(f"    Fitting {model_name_key} with default parameters (no tuning).")
        if is_keras:
            model_instance.fit(X_train, y_train) # KerasRegressor handles its own callbacks
        else: # For XGBoost or other sklearn models, fit directly. XGBoost eval_set handled in train_evaluate_model
            model_instance.fit(X_train, y_train)
        fitted_model = model_instance
    return fitted_model


#--------------------------------------------------------------------------------------------------------------------
#region Helper Functions
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#region Load and Initial Prepare
#--------------------------------------------------------------------------------------------------------------------
print("--- Loading Data ---")
fred = pd.read_parquet('data/fred/macro_data_processed.parquet')
fdic = pd.read_parquet('data/fdic/fdic_data_processed.parquet').set_index(['id', 'date'])
yahoo = pd.read_parquet('data/yahoo/yahoo.parquet')
print("Data loaded.")

print("--- Preparing Data ---")
data_preparer = RegressionDataPreparer(fred, fdic, yahoo, config=c)
print("--- Data prepared ---")


if data_preparer.final_data is None or data_preparer.final_data.empty:
    raise ValueError("Base data preparation resulted in an empty DataFrame. Halting execution.")

all_data_prepared = data_preparer.final_data # For plotting or direct inspection if needed
#endregion

#--------------------------------------------------------------------------------------------------------------------
#region Run regressions
#--------------------------------------------------------------------------------------------------------------------

def train_evaluate_model(model_name_key, model_instance, X_train, y_train, X_test, y_test, 
                         param_grid=None, cv_splitter=None, 
                         use_random_search=False, n_iter_random_search=10):
    """Trains a single model, tunes if param_grid is provided, and evaluates it."""
    if X_train.empty or (hasattr(X_train, 'shape') and X_train.shape[1] == 0):
        print(f"    Skipping {model_name_key}: X_train is empty or has no features.")
        return {'model_object': None, 'predictions_train': None, 'predictions_test': None}

    # Handle XGBoost early stopping eval_set *before* calling _fit_model_with_tuning
    # This is because eval_set is a fit parameter, not a model parameter.
    # Note: Using X_test/y_test for early stopping during final fit can lead to information leakage.
    # A proper validation set split from the training data is preferred for this.
    # Sticking to the current pattern for now, but noting this potential improvement.
    is_xgb = isinstance(model_instance, XGBRegressor)
    is_keras = isinstance(model_instance, KerasRegressor)

    if is_xgb and hasattr(model_instance, 'early_stopping_rounds') and model_instance.early_stopping_rounds is not None:
        if not X_test.empty and not y_test.empty and len(X_test) == len(y_test):
            # Create a new model instance with eval_set configured for the fit call
            # This avoids modifying the original model_instance passed in.
            model_instance_for_fit = model_instance.__class__(**model_instance.get_params())
            model_instance_for_fit.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False) # XGB specific
            fitted_model_to_use = model_instance_for_fit
        else:
            print(f"    XGBoost: Skipping eval_set for early stopping due to empty or misaligned X_test/y_test for {model_name_key}")
            # If eval_set cannot be used, proceed with standard fitting/tuning
            fitted_model_to_use = _fit_model_with_tuning(model_name_key, model_instance, X_train, y_train, param_grid, cv_splitter, use_random_search, n_iter_random_search)
    else:
        # For non-XGBoost models or XGBoost without early stopping
        # For KerasRegressor, early stopping is handled by callbacks passed at instantiation.
        # _fit_model_with_tuning will call KerasRegressor.fit(), which uses these callbacks.
        # If tuning, scikeras uses CV splits for validation with callbacks.
        # If not tuning, KerasRegressor.fit() is called directly; it might use an internal validation_split if configured, or just train loss for early stopping.
        fitted_model_to_use = _fit_model_with_tuning(model_name_key, model_instance, X_train, y_train, param_grid, cv_splitter, use_random_search, n_iter_random_search)

    predictions_train, predictions_test = _predict_and_evaluate(fitted_model_to_use, X_train, y_train, X_test, y_test, model_name_key)

    return {'model_object': fitted_model_to_use, 'predictions_train': predictions_train, 'predictions_test': predictions_test}

def train_evaluate_ensemble(trained_models_dict, X_train_scaled, y_train, X_test_scaled, y_test):
    """Trains and evaluates a VotingRegressor ensemble model."""
    print("    Attempting to train Voting Regressor ensemble...")
    if X_train_scaled.empty or y_train.empty:
        print("    VotingEnsemble: Skipping due to empty training data.")
        return None
    if X_test_scaled.empty or y_test.empty:
        print("    VotingEnsemble: Skipping evaluation due to empty test data.")
        return {'model_object': None, 'predictions_train': None, 'predictions_test': None}
    estimators = [] # List to hold (name, model_blueprint) tuples
    potential_base_models = POTENTIAL_BASE_MODELS_FOR_ENSEMBLE # Use the configured list
    for model_name_key in potential_base_models:
        if model_name_key in trained_models_dict and \
           trained_models_dict[model_name_key] and \
           trained_models_dict[model_name_key].get('model_object') is not None:
            
            original_base_model_obj = trained_models_dict[model_name_key]['model_object']
            
            # Check if the ORIGINAL model was fitted successfully and is suitable
            is_original_suitable_and_fitted = False
            if isinstance(original_base_model_obj, KerasRegressor):
                # For KerasRegressor, if we have the model object from training,
                # assume it's suitable to be cloned and refit by VotingRegressor.
                # A deeper check could be hasattr(original_base_model_obj, 'model_') and original_base_model_obj.model_ is not None
                is_original_suitable_and_fitted = True
            elif hasattr(original_base_model_obj, "coef_") or \
                 hasattr(original_base_model_obj, "feature_importances_"):
                is_original_suitable_and_fitted = True
            elif isinstance(original_base_model_obj, XGBRegressor) and \
                 hasattr(original_base_model_obj, '_Booster') and \
                 original_base_model_obj._Booster is not None: # Check _Booster attribute exists before accessing
                is_original_suitable_and_fitted = True

            if is_original_suitable_and_fitted:
                try: 
                    # Clone the base model to get a blueprint with the right hyperparameters.
                    # VotingRegressor will call fit() on this cloned blueprint.
                    cloned_blueprint = clone(original_base_model_obj)
                    estimators.append((model_name_key.lower(), cloned_blueprint))
                except Exception as e: # Catch potential errors during cloning
                     print(f"    Error cloning model {model_name_key} for ensemble. Skipping. Error: {e}")
            else:
                print(f"    Original model {model_name_key} for ensemble seems not fitted or unsuitable. Skipping.")
    if len(estimators) < 2: 
        print("    Not enough suitable (fitted) base models for Voting Regressor. Skipping ensemble.")
        return None
    try:
        voting_reg = VotingRegressor(estimators=estimators, n_jobs=-1)
        # Ensure data is writeable to avoid "cannot set WRITEABLE flag" error
        X_train_copy = X_train_scaled.copy()
        y_train_copy = y_train.copy()
        voting_reg.fit(X_train_copy, y_train_copy)
        
        predictions_train_ensemble = None
        if not X_train_scaled.empty and len(X_train_scaled) > 0:
            predictions_train_ensemble = voting_reg.predict(X_train_scaled)

        predictions_test_ensemble = None
        if not X_test_scaled.empty and len(X_test_scaled) > 0:
            predictions_test_ensemble = voting_reg.predict(X_test_scaled)
            
        print("    Voting Regressor trained successfully.")
        return {
            'model_object': voting_reg,
            'predictions_train': predictions_train_ensemble,
            'predictions_test': predictions_test_ensemble
        }
    except Exception as e:
        print(f"    Error training Voting Regressor: {e}")
        return None

def _predict_and_evaluate(fitted_model, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series, model_name_key: str):
    """
    Makes predictions using a fitted model and calculates evaluation metrics.
    Returns a dictionary of metrics.
    """
    predictions_train_output = None
    predictions_test_output = None

    if fitted_model is None:
        print(f"    Skipping prediction/evaluation for {model_name_key}: Model is not fitted.")
        return predictions_train_output, predictions_test_output

    if X_test.empty or y_test.empty:
        print(f"    Skipping test prediction/evaluation for {model_name_key} as X_test or y_test is empty.")
    else:
        try:
            predictions_test_output = fitted_model.predict(X_test)
        except Exception as e:
            print(f"    Error during test prediction for {model_name_key}: {e}")

    if not X_train.empty and len(X_train) > 0 and not y_train.empty and len(X_train) == len(y_train):
        try:
            predictions_train_output = fitted_model.predict(X_train)
        except Exception as e_train_rmse:
            print(f"    Could not make train predictions for {model_name_key}: {e_train_rmse}")
    elif not y_train.empty:
         print(f"    Skipping train predictions for {model_name_key} due to empty or mismatched train data (X_train empty: {X_train.empty}, y_train empty: {y_train.empty}, lengths match: {len(X_train) == len(y_train) if not X_train.empty and not y_train.empty else False}).")

    return predictions_train_output, predictions_test_output

#endregion

#--------------------------------------------------------------------------------------------------------------------
#region Plot the data
#--------------------------------------------------------------------------------------------------------------------


def plot_features_timeseries_flat(
    df: pd.DataFrame, 
    date_column: str = None, 
    id_column: str = None, 
    feature_columns: List[str] = None
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if date_column is None: date_column = 'date'
    if id_column is None: id_column = 'id'
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in [date_column, id_column]]
    required_cols = [date_column, id_column] + feature_columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {', '.join(missing_cols)}")
    if not feature_columns:
        print("No feature columns provided to plot.")
        return
    num_features = len(feature_columns)
    num_cols_grid = 3
    num_rows_grid = (num_features + num_cols_grid - 1) // num_cols_grid
    fig, axes = plt.subplots(
        num_rows_grid, num_cols_grid, 
        figsize=(5 * num_cols_grid, 4 * num_rows_grid), 
        sharex=True 
    )
    axes_flat: np.ndarray[plt.Axes]
    if num_rows_grid == 1 and num_cols_grid == 1: axes_flat = np.array([axes])
    elif num_rows_grid == 1 or num_cols_grid == 1: axes_flat = axes if isinstance(axes, np.ndarray) else np.array([axes])
    else: axes_flat = axes.flatten()
    legend_handles_map = {} 
    for i, feature_name in enumerate(feature_columns):
        ax = axes_flat[i]
        sns.lineplot(data=df, x=date_column, y=feature_name, hue=id_column, ax=ax, legend=True)
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_handles_map: legend_handles_map[label] = handle
        if ax.get_legend() is not None: ax.get_legend().remove() 
        ax.set_title(f'{feature_name}')
        ax.set_xlabel(date_column) 
        ax.set_ylabel(feature_name)
        ax.tick_params(axis='x', rotation=45)
    for j in range(num_features, len(axes_flat)): fig.delaxes(axes_flat[j])
    # fig.legend(legend_handles_map.values(), legend_handles_map.keys(), title=id_column, bbox_to_anchor=(1.0, 0.95), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    plt.show()

#endregion



#--------------------------------------------------------------------------------------------------------------------
#region Execute 
#--------------------------------------------------------------------------------------------------------------------
results_store = {}

if SAVE_ARTIFACTS:
    os.makedirs(ARTIFACTS_BASE_DIR, exist_ok=True)
    print(f"Artifacts will be saved in: {os.path.abspath(ARTIFACTS_BASE_DIR)}")

models_config, param_grids_config = get_models_and_param_grids(use_random_search=USE_RANDOM_SEARCH_CV, n_iter_random_search=N_ITER_RANDOM_SEARCH)

for current_target_variable in TARGET_VARIABLES.keys():
    print(f"\n\nProcessing Target Variable: {current_target_variable}")
    results_store[current_target_variable] = {}

    # --- Iterate through each forecast horizon ---
    for i, horizon_val in enumerate(FORECAST_HORIZONS): # Use enumerate to potentially get unique run names if needed
        print(f"\n--- Processing Horizon: {horizon_val}-quarter(s) ahead for Target: {current_target_variable} ---")
        results_store[current_target_variable][horizon_val] = {} 
        
        current_target_artifact_dir_for_horizon = ""
        if SAVE_ARTIFACTS:
            sanitized_target_name = current_target_variable.replace('/', '_').replace('\\', '_')
            target_specific_artifact_dir = os.path.join(ARTIFACTS_BASE_DIR, f"target_{sanitized_target_name}")
            os.makedirs(target_specific_artifact_dir, exist_ok=True)
            current_target_artifact_dir_for_horizon = os.path.join(target_specific_artifact_dir, f"horizon_{horizon_val}")
            os.makedirs(current_target_artifact_dir_for_horizon, exist_ok=True)

        # Prepare data for the current horizon and target using RegressionDataPreparer
        prepared_data = data_preparer.get_horizon_specific_data(horizon=horizon_val, target_variable=current_target_variable)

        if prepared_data is None:
            print(f"  Skipping horizon {horizon_val} for target {current_target_variable} due to data preparation failure.")
            for name_model in models_config.keys():
                results_store[current_target_variable][horizon_val][name_model] = {'model_object': None}
            results_store[current_target_variable][horizon_val]['VotingEnsemble'] = {'model_object': None}
            continue 
                
        X_train_scaled_df, X_test_scaled_df, y_train, y_test, X_train_orig, X_test_orig = prepared_data

        if SAVE_ARTIFACTS and current_target_artifact_dir_for_horizon:
            data_artifact_dir = os.path.join(current_target_artifact_dir_for_horizon, "data")
            os.makedirs(data_artifact_dir, exist_ok=True)
            if X_train_scaled_df is not None and not X_train_scaled_df.empty:
                X_train_scaled_df.to_parquet(os.path.join(data_artifact_dir, "X_train_scaled.parquet"))
            if X_test_scaled_df is not None and not X_test_scaled_df.empty:
                X_test_scaled_df.to_parquet(os.path.join(data_artifact_dir, "X_test_scaled.parquet"))
            if y_train is not None and not y_train.empty:
                y_train_name = y_train.name if y_train.name else f"{current_target_variable}_target_h{horizon_val}_train"
                y_train.to_frame(name=y_train_name).to_parquet(os.path.join(data_artifact_dir, "y_train.parquet"))
            if y_test is not None and not y_test.empty: 
                y_test_name = y_test.name if y_test.name else f"{current_target_variable}_target_h{horizon_val}_test"
                y_test.to_frame(name=y_test_name).to_parquet(os.path.join(data_artifact_dir, "y_test.parquet"))
            if X_train_orig is not None and not X_train_orig.empty: 
                X_train_orig.to_parquet(os.path.join(data_artifact_dir, "X_train_orig.parquet"))
            if X_test_orig is not None and not X_test_orig.empty:
                X_test_orig.to_parquet(os.path.join(data_artifact_dir, "X_test_orig.parquet"))

        current_n_splits_cv = N_SPLITS_CV
        if len(X_train_scaled_df) < current_n_splits_cv + 1 : 
            print(f"  Warning: Training set size ({len(X_train_scaled_df)}) for target {current_target_variable}, H{horizon_val} is too small for {current_n_splits_cv} CV splits. Reducing or skipping CV.")
            if len(X_train_scaled_df) >= 3 and current_n_splits_cv > 1: 
                current_n_splits_cv = max(1, int(len(X_train_scaled_df) / 2) -1) 
                if current_n_splits_cv < 2: current_n_splits_cv = 0 
                print(f"  Reduced CV splits to {current_n_splits_cv}.")
            else:
                print(f"  Skipping CV tuning for target {current_target_variable}, H{horizon_val} due to insufficient training data.")
                current_n_splits_cv = 0 
        tscv_splitter = TimeSeriesSplit(n_splits=current_n_splits_cv) if current_n_splits_cv >= 2 else None

        # Determine which models to run based on MODELS_TO_RUN config
        model_names_to_process_this_run = list(models_config.keys()) # Default to all models
        if MODELS_TO_RUN is not None and MODELS_TO_RUN: # If a specific list is provided and it's not empty
            model_names_to_process_this_run = [
                name for name in MODELS_TO_RUN if name in models_config
            ]
            print(f"  Configured to run only: {model_names_to_process_this_run} for target {current_target_variable}, H{horizon_val}")
            for m_name_cfg in MODELS_TO_RUN:
                if m_name_cfg not in models_config:
                    print(f"  Warning: Model '{m_name_cfg}' in MODELS_TO_RUN is not defined in models_config and will be skipped.")

        for model_name_loop in model_names_to_process_this_run:
            base_model_from_config = models_config[model_name_loop]
            # Always start with a fresh clone for every model in every iteration
            current_model_instance_for_training = clone(base_model_from_config)
            
            print(f"  Training {model_name_loop} for target {current_target_variable}, horizon {horizon_val}...")
            model_results = None 
            if X_train_scaled_df.empty or y_train.empty:
                print(f"    Skipping {model_name_loop} due to empty scaled training data for target {current_target_variable}, H{horizon_val}.")
                results_store[current_target_variable][horizon_val][model_name_loop] = {'model_object': None}
                continue # Skip to the next model in the loop
            
            # If it's a NeuralNetwork, we add specific callbacks to this cloned instance
            if model_name_loop == "NeuralNetwork" and SAVE_ARTIFACTS:
                sanitized_target_name_tb = current_target_variable.replace('/', '_').replace('\\', '_')
                log_dir = os.path.join(ARTIFACTS_BASE_DIR, "tensorboard_logs", f"target_{sanitized_target_name_tb}", f"horizon_{horizon_val}", model_name_loop)
                os.makedirs(log_dir, exist_ok=True)
                tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

                # current_model_instance_for_training is already a clone. Add/modify its callbacks.
                if not hasattr(current_model_instance_for_training, 'callbacks') or current_model_instance_for_training.callbacks is None:
                    current_model_instance_for_training.callbacks = []
                
                # Make a mutable copy of existing callbacks from the cloned instance
                current_callbacks = list(current_model_instance_for_training.callbacks) 
                if not any(isinstance(cb, TensorBoard) for cb in current_callbacks):
                    current_callbacks.append(tensorboard_callback)
                    current_model_instance_for_training.callbacks = current_callbacks # Set the updated list back
                    print(f"    Added TensorBoard logging to: {log_dir} for {model_name_loop}")

            model_results = train_evaluate_model(
                model_name_loop, 
                current_model_instance_for_training,
                X_train_scaled_df, y_train, X_test_scaled_df, y_test,
                param_grid=param_grids_config.get(model_name_loop), 
                cv_splitter=tscv_splitter if param_grids_config.get(model_name_loop) and tscv_splitter else None,
                use_random_search=USE_RANDOM_SEARCH_CV, n_iter_random_search=N_ITER_RANDOM_SEARCH
            )
            if model_results:
                # Store only the model object, predictions are not stored in results_store anymore
                results_store[current_target_variable][horizon_val][model_name_loop] = {
                    'model_object': model_results.get('model_object')
                }
                actual_model_object_from_training = model_results.get('model_object')

                if SAVE_ARTIFACTS and actual_model_object_from_training is not None and current_target_artifact_dir_for_horizon:
                    model_path = os.path.join(current_target_artifact_dir_for_horizon, f"{model_name_loop}.joblib")
                    if model_name_loop == "NeuralNetwork" and hasattr(actual_model_object_from_training, 'model_'):
                        keras_save_path = os.path.join(current_target_artifact_dir_for_horizon, f"{model_name_loop}_keras_model")
                        try:
                            actual_model_object_from_training.model_.save(keras_save_path)
                            print(f"    Saved Keras model for {model_name_loop} to {keras_save_path}")
                        except Exception as e_keras:
                            print(f"    Error saving Keras model for {model_name_loop}: {e_keras}")
                        # Ensure model_object is set to None in results_store if Keras model is saved separately
                        # and the KerasRegressor wrapper itself is not what we want to joblib.
                        # However, if the wrapper *is* what we want, this line is not needed.
                        # For now, assuming we save the Keras model and don't need to joblib the wrapper.
                    else: 
                        try:
                            joblib.dump(actual_model_object_from_training, model_path)
                            print(f"    Saved {model_name_loop} to {model_path}")
                        except Exception as e:
                            print(f"    Error saving {model_name_loop} model: {e}")
                            results_store[current_target_variable][horizon_val][model_name_loop]['model_object'] = None 
            else:
                results_store[current_target_variable][horizon_val][model_name_loop] = {'model_object': None}

        if results_store[current_target_variable][horizon_val] and not (X_train_scaled_df.empty or y_train.empty):
            ensemble_results = train_evaluate_ensemble(results_store[current_target_variable][horizon_val], X_train_scaled_df, y_train, X_test_scaled_df, y_test)
            if ensemble_results:
                # Store only the model object for the ensemble
                results_store[current_target_variable][horizon_val]['VotingEnsemble'] = {
                    'model_object': ensemble_results.get('model_object')
                }
                ensemble_model_object = ensemble_results.get('model_object')
                if SAVE_ARTIFACTS and ensemble_model_object is not None and current_target_artifact_dir_for_horizon:
                    model_path = os.path.join(current_target_artifact_dir_for_horizon, "VotingEnsemble.joblib")
                    try:
                        joblib.dump(ensemble_results['model_object'], model_path)
                        print(f"    Saved VotingEnsemble model to {model_path}")
                    except Exception as e_keras:
                        
                        print(f"    Error saving VotingEnsemble model: {e_keras}")
                        if current_target_variable in results_store and horizon_val in results_store[current_target_variable] and \
                           'VotingEnsemble' in results_store[current_target_variable][horizon_val] and \
                           isinstance(results_store[current_target_variable][horizon_val]['VotingEnsemble'], dict):
                            results_store[current_target_variable][horizon_val]['VotingEnsemble']['model_object'] = None
            else: 
                results_store[current_target_variable][horizon_val]['VotingEnsemble'] = {'model_object': None}
        else: 
            results_store[current_target_variable][horizon_val]['VotingEnsemble'] = {'model_object': None}

    # Feature plotting for target variable, log_total_assets, and deposit_ratio has been removed as per request.

# final_summary_df = aggregate_and_display_results(results_store, plot_result_charts=PLOT_RESULT_CHARTS, print_final_summary=PRINT_FINAL_SUMMARY) # Removed
if SAVE_ARTIFACTS:
    joblib.dump(results_store, os.path.join(ARTIFACTS_BASE_DIR, "results_store.joblib"))
    print(f"Saved full results_store to {os.path.join(ARTIFACTS_BASE_DIR, 'results_store.joblib')}")
#endregion
#endregion


#endregion
