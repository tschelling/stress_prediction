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
from IPython.display import display

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

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Import mean_absolute_error


#--------------------------------------------------------------------------------------------------------------------
#region Configure 
#--------------------------------------------------------------------------------------------------------------------

warnings.filterwarnings('ignore')

# These global variables are used elsewhere in the script (plotting, main loops, artifact naming, etc.)
TARGET_VARIABLE = 'interest_expense_to_assets'
FEATURE_VARIABLES = ['gdp_qoq', 'deposit_ratio', 'loan_to_asset_ratio', 'log_total_assets', 'cpi_qoq', 'unemployment', 
                     'household_delinq', 'tbill_3m', 'tbill_10y', 'spread_10y_3m', 'sp500_qoq', 
                     'corp_bond_spread', 'vix_qoq', 
                     'dep_small_3m_less_to_assets',
                     'dep_small_3m_1y_to_assets',
                     'dep_small_1y_3y_to_assets',
                     'dep_small_3y_more_to_assets',
                     'dep_large_3m_less_to_assets',
                     'dep_large_3m_1y_to_assets', 
                     'dep_large_1y_3y_to_assets',
                     'dep_large_3y_more_to_assets'
                     ]
MODELS_TO_RUN = ["XGBoost", "DecisionTree","LinearRegression", "Lasso", "DummyRegressor", "RandomForest"] # Options: None (all models), or a list of model names, e.g., ["XGBoost", "Ridge", "NeuralNetwork"]
# MODELS_TO_RUN = ["XGBoost", "Ridge", "NeuralNetwork", "DummyRegressor"] # Example: run only these
# MODELS_TO_RUN = ["XGBoost"] # Example: run only XGBoost

FORECAST_HORIZONS = list(range(1, 2))
N_SPLITS_CV = 3

# Suppress TensorFlow INFO and WARNING messages for cleaner output
tf.get_logger().setLevel('ERROR')

# Enable Mixed Precision for TensorFlow on compatible GPUs (like M4 Metal)
if tf.config.list_physical_devices('GPU'):
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print("TensorFlow Mixed Precision policy set to 'mixed_float16'.")

# Training parameters
USE_RANDOM_SEARCH_CV = True 
N_ITER_RANDOM_SEARCH = 100
 
# Artifact Storage
SAVE_ARTIFACTS = True
ARTIFACTS_BASE_DIR = "model_run_artifacts_test3"

# Display
PLOT_RESULT_CHARTS = False
PRINT_FINAL_SUMMARY = True

c = {
    'TARGET_VARIABLE': TARGET_VARIABLE,                 
    'FEATURE_VARIABLES': FEATURE_VARIABLES,             
    'INCLUDE_TIME_FE': True,                           
    'INCLUDE_BANK_FE': False,                            
    'OUTLIER_THRESHOLD_TARGET': 3.0,                    
    'MIN_OBS_PER_BANK': 10,                             
    'DATA_BEGIN': '2017-01-01',                              
    'DATA_END': None,                                  
    'RESTRICT_TO_NUMBER_OF_BANKS': 500,                 
    'RESTRICT_TO_BANK_SIZE': None,                      
    'RESTRICT_TO_MINIMAL_DEPOSIT_RATIO': None,          
    'RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO': None,     
    'INCLUDE_AUTOREGRESSIVE_LAGS': True,                
    'NUMBER_OF_LAGS_TO_INCLUDE': 4,                     
    'TRAIN_TEST_SPLIT_DIMENSION': 'date',               
    'TEST_SPLIT': 0.2                                  # Takes a number or a date in the format 'YYYY-MM-DD'
}

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
    from scipy.stats import uniform, randint

    models = {
        "DummyRegressor": sklearn.dummy.DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(random_state=42, max_iter=15000),
        "Ridge": Ridge(random_state=42, max_iter=15000),
        "ElasticNet": ElasticNet(random_state=42, max_iter=15000),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1, tree_method='hist'), # Removed enable_categorical
        "RandomForest": RandomForestRegressor(
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
        )
    }
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

def _calculate_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict:
    """Calculates standard regression metrics."""
    if y_true.empty or len(predictions) != len(y_true):
        return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan}

    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions)

    # Calculate MAPE, handling potential division by zero
    y_true_no_zeros = y_true.replace(0, np.nan).dropna() # Use np.nan for robustness
    if not y_true_no_zeros.empty:
        predictions_aligned = pd.Series(predictions, index=y_true.index).loc[y_true_no_zeros.index]
        mape = np.mean(np.abs((y_true_no_zeros - predictions_aligned) / y_true_no_zeros)) * 100
    else:
        mape = np.nan
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


#--------------------------------------------------------------------------------------------------------------------
#region Load and Initial Prepare
#--------------------------------------------------------------------------------------------------------------------
print("--- Loading Data ---")
data_precleaning = pd.read_parquet('data.parquet')
print(f"Raw data loaded. Shape: {data_precleaning.shape}")

# Configuration dictionary for RegressionDataPreparer
# Values that were previously global variables specific to data prep are now defined here.


data_preparer = RegressionDataPreparer(initial_df=data_precleaning, config=c)

if data_preparer.base_data_for_horizons is None or data_preparer.base_data_for_horizons.empty:
    raise ValueError("Base data preparation resulted in an empty DataFrame. Halting execution.")

all_data_prepared = data_preparer.base_data_for_horizons # For plotting or direct inspection if needed
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
        return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': model_instance}

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

    # Use the new evaluation function
    metrics = _predict_and_evaluate(fitted_model_to_use, X_train, y_train, X_test, y_test, model_name_key)

    # Add the fitted model object to the metrics dictionary
    metrics['model_object'] = fitted_model_to_use

    return metrics

def train_evaluate_ensemble(trained_models_dict, X_train_scaled, y_train, X_test_scaled, y_test):
    """Trains and evaluates a VotingRegressor ensemble model."""
    print("    Attempting to train Voting Regressor ensemble...")
    if X_train_scaled.empty or y_train.empty:
        print("    VotingEnsemble: Skipping due to empty training data.")
        return None
    if X_test_scaled.empty or y_test.empty:
        print("    VotingEnsemble: Skipping evaluation due to empty test data.")
        return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
    estimators = []
    potential_base_models = ["XGBoost", "Ridge", "ElasticNet", "NeuralNetwork"] # Added NeuralNetwork
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
        voting_reg.fit(X_train_scaled, y_train)
        predictions_test = voting_reg.predict(X_test_scaled)
        rmse_train_ensemble = np.nan
        if not X_train_scaled.empty and len(X_train_scaled) > 0:
            predictions_train = voting_reg.predict(X_train_scaled)
            if len(predictions_train) == len(y_train): 
                 rmse_train_ensemble = np.sqrt(mean_squared_error(y_train, predictions_train))
            else:
                print(f"    VotingEnsemble WARNING: Train prediction length ({len(predictions_train)}) mismatch with y_train ({len(y_train)}).")
        test_metrics = _calculate_metrics(y_test, predictions_test)
        print("    Voting Regressor trained successfully.")
        return {
            **test_metrics, 
            'RMSE_train': rmse_train_ensemble,
            'model_object': voting_reg
        }
    except Exception as e:
        print(f"    Error training Voting Regressor: {e}")
        return None

def aggregate_and_display_results(results_store: dict, plot_result_charts: bool = False, print_final_summary: bool = True):
    """Aggregates results from all horizons and models, prints a summary, and plots RMSE."""
    print("\n\n--- Final Model Performance Summary ---")
    summary_entries = []
    train_test_rmse_entries = []
    for horizon, models_results in results_store.items():
        for model_name, metrics in models_results.items():
            if isinstance(metrics, dict):
                rmse_test_val = metrics.get('RMSE', np.nan)
                rmse_train_val = metrics.get('RMSE_train', np.nan)
                summary_entries.append({
                    'Horizon': horizon,
                    'Model': model_name,
                    'MAE': metrics.get('MAE', np.nan),
                    'RMSE': rmse_test_val,
                    'R2': metrics.get('R2', np.nan),
                    'MAPE': metrics.get('MAPE', np.nan)
                })
                train_test_rmse_entries.append({
                    'Horizon': horizon,
                    'Model': model_name,
                    'RMSE_Train': rmse_train_val,
                    'RMSE_Test': rmse_test_val
                })
            else: 
                 summary_entries.append({
                    'Horizon': horizon,
                    'Model': model_name, 
                    'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan
                })
                 train_test_rmse_entries.append({
                    'Horizon': horizon,
                    'Model': model_name,
                    'RMSE_Train': np.nan,
                    'RMSE_Test': np.nan
                })
    if not summary_entries:
        print("No results to display. Check for errors during model training or evaluation.")
        return pd.DataFrame() 
    summary_df = pd.DataFrame(summary_entries)
    # Ensure RMSE is numeric before sorting
    summary_df = summary_df.sort_values(by=['Horizon', 'RMSE'])
    if print_final_summary:
        print("\nOverall Performance Metrics (Test Set):")
        display(summary_df) 
    if plot_result_charts:
        if summary_df.empty or not FORECAST_HORIZONS or summary_df['RMSE'].isnull().all():
            print("Cannot plot results: summary is empty, RMSE is all NaN, or FORECAST_HORIZONS not defined.")
            return summary_df
        plt.figure(figsize=(14, 8))
        unique_models = summary_df['Model'].unique()
        for model_name_plot in unique_models: 
            model_data = summary_df[summary_df['Model'] == model_name_plot].dropna(subset=['RMSE', 'Horizon']) 
            if not model_data.empty:
                 plt.plot(model_data['Horizon'].astype(str), model_data['RMSE'], marker='o', linestyle='-', label=model_name_plot) 
        plt.xlabel("Forecast Horizon (Quarters)")
        plt.ylabel("Root Mean Squared Error (RMSE)")
        plt.title("Model RMSE vs. Forecast Horizon", fontsize=16)
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout() 
        plt.show()
    if train_test_rmse_entries and print_final_summary:
        train_test_rmse_df = pd.DataFrame(train_test_rmse_entries)
        train_test_rmse_df = train_test_rmse_df.sort_values(by=['Horizon', 'RMSE_Test'])
        print("\n\n--- Training vs. Test RMSE Comparison ---")
        display(train_test_rmse_df)
    return summary_df

def _predict_and_evaluate(fitted_model, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series, model_name_key: str):
    """
    Makes predictions using a fitted model and calculates evaluation metrics.
    Returns a dictionary of metrics.
    """
    metrics = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan}

    if fitted_model is None:
        print(f"    Skipping prediction/evaluation for {model_name_key}: Model is not fitted.")
        return metrics

    if X_test.empty or y_test.empty:
        print(f"    Skipping test prediction/evaluation for {model_name_key} as X_test or y_test is empty.")
    else:
        try:
            predictions_test = fitted_model.predict(X_test)
            metrics.update(_calculate_metrics(y_test, predictions_test))
        except Exception as e:
            print(f"    Error during test prediction for {model_name_key}: {e}")

    # Calculate train RMSE (optional, for checking overfitting)
    if not X_train.empty and len(X_train) > 0 and not y_train.empty and len(X_train) == len(y_train):
        try:
            predictions_train = fitted_model.predict(X_train)
            metrics['RMSE_train'] = np.sqrt(mean_squared_error(y_train, predictions_train)) 
        except Exception as e_train_rmse:
            print(f"    Could not calculate train RMSE for {model_name_key}: {e_train_rmse}")
            metrics['RMSE_train'] = np.nan
    elif not y_train.empty:
         print(f"    Skipping train RMSE for {model_name_key} due to empty or mismatched train data (X_train empty: {X_train.empty}, y_train empty: {y_train.empty}, lengths match: {len(X_train) == len(y_train) if not X_train.empty and not y_train.empty else False}).")

    return metrics

#endregion




#--------------------------------------------------------------------------------------------------------------------
#region Execute 
#--------------------------------------------------------------------------------------------------------------------
results_store = {}

if SAVE_ARTIFACTS:
    os.makedirs(ARTIFACTS_BASE_DIR, exist_ok=True)
    print(f"Artifacts will be saved in: {os.path.abspath(ARTIFACTS_BASE_DIR)}")

models_config, param_grids_config = get_models_and_param_grids(use_random_search=USE_RANDOM_SEARCH_CV, n_iter_random_search=N_ITER_RANDOM_SEARCH)

# --- Iterate through each forecast horizon ---
for i, horizon_val in enumerate(FORECAST_HORIZONS): # Use enumerate to potentially get unique run names if needed
    print(f"\n--- Processing Horizon: {horizon_val}-quarter(s) ahead ---")
    results_store[horizon_val] = {} 
    
    current_horizon_artifact_dir = ""
    if SAVE_ARTIFACTS:
        current_horizon_artifact_dir = os.path.join(ARTIFACTS_BASE_DIR, f"horizon_{horizon_val}")
        os.makedirs(current_horizon_artifact_dir, exist_ok=True)

    # Prepare data for the current horizon using RegressionDataPreparer
    prepared_data = data_preparer.get_horizon_specific_data(horizon=horizon_val)

    if prepared_data is None:
        print(f"  Skipping horizon {horizon_val} due to data preparation failure.")
        for name_model in models_config.keys():
            results_store[horizon_val][name_model] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
        results_store[horizon_val]['VotingEnsemble'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
        continue 
            
    X_train_scaled_df, X_test_scaled_df, y_train, y_test, X_train_orig, X_test_orig = prepared_data

    if SAVE_ARTIFACTS:
        data_artifact_dir = os.path.join(current_horizon_artifact_dir, "data")
        os.makedirs(data_artifact_dir, exist_ok=True)
        if X_train_scaled_df is not None and not X_train_scaled_df.empty:
            X_train_scaled_df.to_parquet(os.path.join(data_artifact_dir, "X_train_scaled.parquet"))
        if X_test_scaled_df is not None and not X_test_scaled_df.empty:
            X_test_scaled_df.to_parquet(os.path.join(data_artifact_dir, "X_test_scaled.parquet"))
        if y_train is not None and not y_train.empty:
            y_train_name = y_train.name if y_train.name else TARGET_VARIABLE + f"_target_h{horizon_val}_train"
            y_train.to_frame(name=y_train_name).to_parquet(os.path.join(data_artifact_dir, "y_train.parquet"))
        if y_test is not None and not y_test.empty: 
            y_test_name = y_test.name if y_test.name else TARGET_VARIABLE + f"_target_h{horizon_val}_test"
            y_test.to_frame(name=y_test_name).to_parquet(os.path.join(data_artifact_dir, "y_test.parquet"))
        if X_train_orig is not None and not X_train_orig.empty: 
            X_train_orig.to_parquet(os.path.join(data_artifact_dir, "X_train_orig.parquet"))
        if X_test_orig is not None and not X_test_orig.empty:
            X_test_orig.to_parquet(os.path.join(data_artifact_dir, "X_test_orig.parquet"))

    current_n_splits_cv = N_SPLITS_CV
    if len(X_train_scaled_df) < current_n_splits_cv + 1 : 
        print(f"  Warning: Training set size ({len(X_train_scaled_df)}) is too small for {current_n_splits_cv} CV splits. Reducing or skipping CV.")
        if len(X_train_scaled_df) >= 3 and current_n_splits_cv > 1: 
            current_n_splits_cv = max(1, int(len(X_train_scaled_df) / 2) -1) 
            if current_n_splits_cv < 2: current_n_splits_cv = 0 
            print(f"  Reduced CV splits to {current_n_splits_cv}.")
        else:
            print(f"  Skipping CV tuning for horizon {horizon_val} due to insufficient training data.")
            current_n_splits_cv = 0 
    tscv_splitter = TimeSeriesSplit(n_splits=current_n_splits_cv) if current_n_splits_cv >= 2 else None

    # Determine which models to run based on MODELS_TO_RUN config
    model_names_to_process_this_horizon = list(models_config.keys()) # Default to all models
    if MODELS_TO_RUN is not None and MODELS_TO_RUN: # If a specific list is provided and it's not empty
        model_names_to_process_this_horizon = [
            name for name in MODELS_TO_RUN if name in models_config
        ]
        # Log which models are being skipped or included based on the config
        print(f"  Configured to run only: {model_names_to_process_this_horizon}")
        for m_name_cfg in MODELS_TO_RUN:
            if m_name_cfg not in models_config:
                print(f"  Warning: Model '{m_name_cfg}' in MODELS_TO_RUN is not defined in models_config and will be skipped.")

    for model_name_loop in model_names_to_process_this_horizon:
        model_instance_loop = models_config[model_name_loop]
        print(f"  Training {model_name_loop} for horizon {horizon_val}...")
        model_results = None 
        if X_train_scaled_df.empty or y_train.empty:
            print(f"    Skipping {model_name_loop} due to empty scaled training data for horizon {horizon_val}.")
            results_store[horizon_val][model_name_loop] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
            continue # Skip to the next model in the loop

        current_model_instance_for_training = model_instance_loop # Default to original instance
        # Special handling for NeuralNetwork to add TensorBoard callback before training/tuning
        if model_name_loop == "NeuralNetwork":
            log_dir = os.path.join(ARTIFACTS_BASE_DIR, "tensorboard_logs", f"horizon_{horizon_val}", model_name_loop)
            os.makedirs(log_dir, exist_ok=True)
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Clone the KerasRegressor instance to avoid modifying the original in models_config
            cloned_nn_wrapper = clone(model_instance_loop) 
            
            # Ensure callbacks attribute exists and is a list
            if not hasattr(cloned_nn_wrapper, 'callbacks') or cloned_nn_wrapper.callbacks is None:
                cloned_nn_wrapper.callbacks = []
            
            # Make a mutable copy of existing callbacks (e.g., EarlyStopping from initial config)
            current_callbacks = list(cloned_nn_wrapper.callbacks) 
            
            # Add TensorBoard callback if a TensorBoard callback isn't already there
            if not any(isinstance(cb, TensorBoard) for cb in current_callbacks):
                current_callbacks.append(tensorboard_callback)
                cloned_nn_wrapper.callbacks = current_callbacks # Set the updated list back
                print(f"    Added TensorBoard logging to: {log_dir} for {model_name_loop}")
            
            current_model_instance_for_training = cloned_nn_wrapper # Use the cloned and modified instance

        model_results = train_evaluate_model(
            model_name_loop, 
            current_model_instance_for_training, # Pass the original or (for NN) cloned-and-modified instance
            X_train_scaled_df, y_train, X_test_scaled_df, y_test,
            param_grid=param_grids_config.get(model_name_loop), 
            cv_splitter=tscv_splitter if param_grids_config.get(model_name_loop) and tscv_splitter else None,
            use_random_search=USE_RANDOM_SEARCH_CV, n_iter_random_search=N_ITER_RANDOM_SEARCH
        )
        if model_results:
            # Store the full results including the model_object initially
            results_store[horizon_val][model_name_loop] = model_results 
            
            actual_model_object_from_training = model_results.get('model_object')

            if SAVE_ARTIFACTS and actual_model_object_from_training is not None:
                model_path = os.path.join(current_horizon_artifact_dir, f"{model_name_loop}.joblib")
                if model_name_loop == "NeuralNetwork" and hasattr(actual_model_object_from_training, 'model_'):
                    # Keras models should be saved using their own save method
                    keras_save_path = os.path.join(current_horizon_artifact_dir, f"{model_name_loop}_keras_model")
                    try:
                        actual_model_object_from_training.model_.save(keras_save_path)
                        print(f"    Saved Keras model for {model_name_loop} to {keras_save_path}")
                        # Note: The KerasRegressor wrapper itself might not be joblib-pickleable.
                        # We won't attempt to joblib.dump the wrapper here to avoid the known error.
                    except Exception as e_keras:
                        print(f"    Error saving Keras model for {model_name_loop}: {e_keras}")
                    # Even if Keras saving fails, we still want to remove the unpickleable object
                    # from results_store to allow the main results_store.joblib dump to work.
                    # For NN, always set model_object to None in results_store for the main joblib dump.
                    results_store[horizon_val][model_name_loop]['model_object'] = None
                else: # For all other models, use joblib
                    try:
                        joblib.dump(actual_model_object_from_training, model_path)
                        print(f"    Saved {model_name_loop} to {model_path}")
                        # If joblib dump succeeds, the model_object REMAINS in results_store
                    except Exception as e:
                        print(f"    Error saving {model_name_loop} model: {e}")
                        results_store[horizon_val][model_name_loop]['model_object'] = None # Set to None only if save fails
        else:
            results_store[horizon_val][model_name_loop] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}

    if results_store[horizon_val] and not (X_train_scaled_df.empty or y_train.empty):
        ensemble_results = train_evaluate_ensemble(results_store[horizon_val], X_train_scaled_df, y_train, X_test_scaled_df, y_test)
        if ensemble_results:
            results_store[horizon_val]['VotingEnsemble'] = ensemble_results
            if SAVE_ARTIFACTS and ensemble_results.get('model_object') is not None:
                model_path = os.path.join(current_horizon_artifact_dir, "VotingEnsemble.joblib")
                try:
                    joblib.dump(ensemble_results['model_object'], model_path)
                    print(f"    Saved VotingEnsemble model to {model_path}")
                except Exception as e:
                    print(f"    Error saving VotingEnsemble model: {e}")
                    if horizon_val in results_store and 'VotingEnsemble' in results_store[horizon_val] and isinstance(results_store[horizon_val]['VotingEnsemble'], dict):
                        results_store[horizon_val]['VotingEnsemble']['model_object'] = None
        else: 
            results_store[horizon_val]['VotingEnsemble'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
    else: 
        results_store[horizon_val]['VotingEnsemble'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}

final_summary_df = aggregate_and_display_results(results_store, plot_result_charts=PLOT_RESULT_CHARTS, print_final_summary=PRINT_FINAL_SUMMARY)
if SAVE_ARTIFACTS:
    joblib.dump(results_store, os.path.join(ARTIFACTS_BASE_DIR, "results_store.joblib"))
    print(f"Saved full results_store to {os.path.join(ARTIFACTS_BASE_DIR, 'results_store.joblib')}")
    if final_summary_df is not None and not final_summary_df.empty:
        final_summary_df.to_csv(os.path.join(ARTIFACTS_BASE_DIR, "final_summary_metrics.csv"), index=False)
        print(f"Saved final_summary_df to {os.path.join(ARTIFACTS_BASE_DIR, 'final_summary_metrics.csv')}")
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

if all_data_prepared is not None and not all_data_prepared.empty:
    # Ensure TARGET_VARIABLE and other desired features for plotting are in all_data_prepared.columns
    plot_cols = [TARGET_VARIABLE]
    if 'log_total_assets' in all_data_prepared.columns: plot_cols.append('log_total_assets')
    if 'deposit_ratio' in all_data_prepared.columns: plot_cols.append('deposit_ratio')
    
    # Filter to only existing columns to avoid errors
    plot_cols_existing = [col for col in plot_cols if col in all_data_prepared.columns]
    if plot_cols_existing:
        plot_features_timeseries_flat(all_data_prepared.reset_index(), feature_columns=plot_cols_existing)
    else:
        print("Warning: None of the specified columns for timeseries plot exist in all_data_prepared.")
else:
    print("Skipping feature timeseries plot as all_data_prepared is empty or None.")
#endregion




#--------------------------------------------------------------------------------------------------------------------
#region Plot estimated vs actual values
#--------------------------------------------------------------------------------------------------------------------
import math # Ensure math is imported if used

results_h1 = results_store.get(1, {}) 
X_train_scaled_df_h1, X_test_scaled_df_h1, y_train_h1, y_test_h1 = None, None, None, None
h1_data_loaded_successfully = False
if 1 in FORECAST_HORIZONS: 
    if SAVE_ARTIFACTS and ARTIFACTS_BASE_DIR:
        h1_data_dir = os.path.join(ARTIFACTS_BASE_DIR, "horizon_1", "data")
        required_files = {
            "X_train": os.path.join(h1_data_dir, "X_train_scaled.parquet"),
            "X_test": os.path.join(h1_data_dir, "X_test_scaled.parquet"),
            "y_train": os.path.join(h1_data_dir, "y_train.parquet"),
            "y_test": os.path.join(h1_data_dir, "y_test.parquet")
        }
        if all(os.path.exists(p) for p in required_files.values()):
            try:
                X_train_scaled_df_h1 = pd.read_parquet(required_files["X_train"])
                X_test_scaled_df_h1 = pd.read_parquet(required_files["X_test"])
                y_train_df_h1 = pd.read_parquet(required_files["y_train"])
                y_test_df_h1 = pd.read_parquet(required_files["y_test"])
                y_train_h1 = y_train_df_h1[y_train_df_h1.columns[0]]
                y_test_h1 = y_test_df_h1[y_test_df_h1.columns[0]]
                h1_data_loaded_successfully = True
                print("Successfully loaded Horizon 1 data from artifacts for plotting.")
            except Exception as e:
                print(f"Error loading Horizon 1 data from artifacts: {e}. Plotting may be affected.")
        else:
            print("Warning: Not all Horizon 1 data artifacts found. Plotting may use last horizon's data or fail if it was not H1.")
    if not h1_data_loaded_successfully:
        if 'X_train_scaled_df' in locals() and isinstance(X_train_scaled_df, pd.DataFrame): 
            X_train_scaled_df_h1 = X_train_scaled_df
            X_test_scaled_df_h1 = X_test_scaled_df
            y_train_h1 = y_train
            y_test_h1 = y_test
            print("Using data from last processed iteration for Horizon 1 plotting (fallback).")
        else:
            print("Error: Horizon 1 data not available for plotting. Globals not set or artifacts failed.")
models_to_plot = []
for model_name, metrics in results_h1.items():
    if 'model_object' in metrics and metrics['model_object'] is not None and hasattr(metrics['model_object'], 'predict'):
        # Removed check for Prophet
            models_to_plot.append((model_name, metrics))
if not models_to_plot:
    print("No suitable models found in results_h1 to plot, or H1 data is unavailable.")
else:
    models_to_plot_count = len(models_to_plot)
    num_cols = 3
    num_rows = math.ceil(models_to_plot_count / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows), squeeze=False)
    axes = axes.flatten()
    plot_index = 0
    def plot_aggregated_timeseries(ax, data_series, label_prefix, color, line_style='-'):
        if data_series is None or data_series.empty:
            print(f"Skipping {label_prefix} plot: data is empty or None.")
            return
        if not isinstance(data_series.index, pd.MultiIndex) or 'date' not in data_series.index.names:
            print(f"Skipping {label_prefix} plot: data index is not a MultiIndex with 'date'. Index type: {type(data_series.index)}")
            return
        quantiles_series = data_series.groupby(level='date').quantile([0.25, 0.50, 0.75])
        if quantiles_series.empty:
            print(f"Warning: Empty quantiles for {label_prefix}. Skipping plot for it.")
            return
        quantiles_df = quantiles_series.unstack()
        if not (0.25 in quantiles_df.columns and 0.50 in quantiles_df.columns and 0.75 in quantiles_df.columns):
            print(f"Warning: Could not find all quantile columns (0.25, 0.50, 0.75) for {label_prefix}. Columns: {quantiles_df.columns}. Skipping.")
            return
        dates_idx = quantiles_df.index
        mean_values = quantiles_df[0.50]
        q1_values = quantiles_df[0.25]
        q3_values = quantiles_df[0.75]
        ax.plot(dates_idx, mean_values, label=f'{label_prefix} Mean', color=color, linestyle=line_style)
        ax.fill_between(dates_idx, q1_values, q3_values, color=color, alpha=0.2, label=f'{label_prefix} IQR')
    for model_name, metrics in models_to_plot:
        ax = axes[plot_index]
        model = metrics['model_object']
        mape = metrics.get('MAPE', float('nan'))
        empty_mi_train = pd.MultiIndex.from_tuples([], names=['id', 'date']) if y_train_h1 is None else y_train_h1.index
        empty_mi_test = pd.MultiIndex.from_tuples([], names=['id', 'date']) if y_test_h1 is None else y_test_h1.index
        predictions_train_series = pd.Series(dtype='float64', index=empty_mi_train)
        predictions_test_series = pd.Series(dtype='float64', index=empty_mi_test)
        if X_train_scaled_df_h1 is not None and not X_train_scaled_df_h1.empty and \
           y_train_h1 is not None and not y_train_h1.empty:
            if X_train_scaled_df_h1.shape[1] > 0: 
                raw_predictions_train = model.predict(X_train_scaled_df_h1)
                predictions_train_series = pd.Series(raw_predictions_train, index=y_train_h1.index)
            else:
                print(f"Skipping train predictions for {model_name} (H1): X_train_scaled_df_h1 has no features.")
        if X_test_scaled_df_h1 is not None and not X_test_scaled_df_h1.empty and \
           y_test_h1 is not None and not y_test_h1.empty:
            if X_test_scaled_df_h1.shape[1] > 0: 
                raw_predictions_test = model.predict(X_test_scaled_df_h1)
                predictions_test_series = pd.Series(raw_predictions_test, index=y_test_h1.index)
            else:
                print(f"Skipping test predictions for {model_name} (H1): X_test_scaled_df_h1 has no features.")
        plot_aggregated_timeseries(ax, y_train_h1, 'Actual Train', 'blue')
        plot_aggregated_timeseries(ax, predictions_train_series, 'Predicted Train', 'orange', line_style='--')
        plot_aggregated_timeseries(ax, y_test_h1, 'Actual Test', 'green')
        plot_aggregated_timeseries(ax, predictions_test_series, 'Predicted Test', 'red', line_style='--')
        ax.set_title(f'{model_name} (H=1) (MAPE: {mape:.2f}%)', fontsize=10)
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel(TARGET_VARIABLE, fontsize=8)
        if (y_train_h1 is not None and not y_train_h1.empty) or \
           (y_test_h1 is not None and not y_test_h1.empty):
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.7)
        plot_index += 1
    for i in range(plot_index, len(axes)): fig.delaxes(axes[i])
    plt.tight_layout(pad=2.0)
    fig.suptitle("Model Predictions vs Actuals (Horizon 1 - Mean and IQR over Banks)", fontsize=16, y=0.995)
    plt.show()
#endregion




#--------------------------------------------------------------------------------------------------------------------
#region Feature importance plotting
#--------------------------------------------------------------------------------------------------------------------
def plot_feature_importance(model, feature_names, ax=None, top_n=10):
    """Plots feature importance for tree-based models or coefficients for linear models."""
    if ax is None:
        fig_fi, ax = plt.subplots(figsize=(10, 6)) # Renamed fig to fig_fi
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1] 
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_feature_names = np.array(feature_names)[top_indices]
        ax.bar(range(len(top_importances)), top_importances, align='center')
        ax.set_xticks(range(len(top_importances)))
        ax.set_xticklabels(top_feature_names, rotation=45, ha='right')
        ax.set_title('Feature Importances')
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        indices = np.argsort(np.abs(coef))[::-1]  
        top_indices = indices[:top_n]
        top_coef = coef[top_indices]
        top_feature_names = np.array(feature_names)[top_indices]
        ax.bar(range(len(top_coef)), top_coef, align='center')
        ax.set_xticks(range(len(top_coef)))
        ax.set_xticklabels(top_feature_names, rotation=45, ha='right')
        ax.set_title('Feature Coefficients')
    ax.set_ylabel('Importance / Coefficient Value')
    plt.tight_layout()
    # plt.show() # Removed to allow fig_fi to be shown if ax is None initially

# Use X_train_scaled_df_h1 for feature names
if 1 in results_store and \
   'XGBoost' in results_store[1] and \
   results_store[1].get('XGBoost') and \
   results_store[1]['XGBoost'].get('model_object') is not None and \
   X_train_scaled_df_h1 is not None and not X_train_scaled_df_h1.empty:
    model_to_plot_fi = results_store[1]['XGBoost']['model_object']
    actual_feature_names = X_train_scaled_df_h1.columns.tolist()
    fig_fi_main, ax_fi_main = plt.subplots(figsize=(10,6)) # Create fig and ax here
    plot_feature_importance(
        model_to_plot_fi,
        actual_feature_names,
        ax=ax_fi_main, 
        top_n=10 
    )
    plt.show() # Show the plot after calling the function
else:
    print("Could not plot feature importance for XGBoost (H=1): Model, its training data, or feature names not available.")
#endregion
