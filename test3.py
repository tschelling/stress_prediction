
import pandas as pd
import numpy as np
import warnings


#region Configure 

warnings.filterwarnings('ignore')

# --- 0. Configuration & Helper Functions ---
TARGET_VARIABLE = 'interest_income_to_assets'
FEATURE_VARIABLES = ['interest_income_to_assets', 'gdp_qoq', 'deposit_ratio', 'loan_to_asset_ratio', 'log_total_assets', 'cpi_qoq', 'unemployment', 
                     'household_delinq', 'tbill_3m', 'tbill_10y', 'spread_10y_3m']
FORECAST_HORIZONS = list(range(1, 2))  # 1 to 8 quarters ahead
NUMBER_OF_LAGS_TO_INCLUDE = 4
# Test split ratio
TRAIN_TEST_SPLIT_DIMENSION = 'date'
TEST_SPLIT_RATIO = 0.2
N_SPLITS_CV = 3

# Data cut offs
DATA_BEGIN = '2000-01-01'  # Start date for the data
DATA_END = '2009-01-01'  # End date for the data

# Cleaning parameters
RESTRICT_TO_NUMBER_OF_BANKS = 200  # If a number, restrict to this many banks
RESTRICT_TO_BANK_SIZE = None  # If a number, restrict to banks with total assets >= this value
RESTRICT_TO_MINIMAL_DEPOSIT_RATIO = 0.5 #0.5  # If a number, restrict to banks with deposit ratio >= this value
RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO = 0.1  # If a number, restrict to banks with change in deposit ratio <= this value

# Regression parameters
INCLUDE_TIME_FE = False  # Include time features in the regression
INCLUDE_BANK_FE = False  # Include unit features in the regression, i.e. bank identifiers

# Training parameters
USE_RANDOM_SEARCH_CV = True # Set to True to use RandomizedSearchCV, False for GridSearchCV
N_ITER_RANDOM_SEARCH = 50   # Number of parameter settings that are sampled for RandomizedSearchCV

# Artifact Storage
SAVE_ARTIFACTS = True
ARTIFACTS_BASE_DIR = "model_run_artifacts_test3" # Changed to avoid conflict if test4.py also uses this

def get_models_and_param_grids(use_random_search=False, n_iter_random_search=10): # Added n_iter
    """Defines models and their hyperparameter grids for GridSearchCV."""
    models = {
        "DummyRegressor": sklearn.dummy.DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(random_state=42, max_iter=15000), # Increased max_iter
        "Ridge": Ridge(random_state=42, max_iter=15000), # Increased max_iter
        "ElasticNet": ElasticNet(random_state=42, max_iter=15000), # Increased max_iter
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1,
                                early_stopping_rounds=None, enable_categorical = True),
        "FixedEffectsLR": LinearRegression() # Added Fixed Effects Linear Regression
    }
    if use_random_search:
        param_grids = {
            "Lasso": {'alpha': uniform(0.0001, 10 - 0.0001)}, # Max is exclusive for uniform
            "Ridge": {'alpha': uniform(0.001, 100 - 0.001)},
            "ElasticNet": {
                'alpha': uniform(0.001, 10 - 0.001),
                'l1_ratio': uniform(0.1, 0.9 - 0.1) # e.g. 0.1 to 0.9
            },
            "DecisionTree": {
                'max_depth': [None] + list(randint(3, 21).rvs(size=min(5, n_iter_random_search // 2), random_state=42)), # Ensure size is reasonable
                'min_samples_split': randint(2, 21),
                'min_samples_leaf': randint(1, 11)
            },
            "XGBoost": {
                'n_estimators': randint(10, 151),
                'learning_rate': uniform(0.01, 0.3 - 0.01),
                'max_depth': randint(1, 12),
                'subsample': uniform(0.6, 1.0 - 0.6),
                'colsample_bytree': uniform(0.6, 1.0 - 0.6)
            }
        }
    else: # Your original GridSearchCV-style grids
        param_grids = {
            "Lasso": {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 20]},
            "Ridge": {'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100]},
            "ElasticNet": {'alpha': [0.001, 0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
            "DecisionTree": {'max_depth': [None, 5, 10, 15, 20], 'min_samples_split': [2, 5, 10, 20],
                             'min_samples_leaf': [1, 5, 10]},
            "XGBoost": {
                'n_estimators': [20, 30, 35, 40, 45, 50, 100],
                'learning_rate': [0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.1, 0.2],
                'max_depth': [1, 2, 3, 5, 7, 9, 11],
            }
        }
    return models, param_grids


# Display
PLOT_RESULT_CHARTS = False
PRINT_FINAL_SUMMARY = True
#endregion





#region Load

data_precleaning = pd.read_parquet('data.parquet')

# Drop observations after a certain date
# df = df[df.index <= '2011-12-31']

# Add time and unit features if specified
if INCLUDE_TIME_FE:
    data_precleaning['quarter'] = 'quarter_' + data_precleaning.index.get_level_values('date').quarter.astype(str)
    data_precleaning['year'] = 'year_' + data_precleaning.index.get_level_values('date').year.astype(str)
if INCLUDE_BANK_FE:
    # 'id' consists of bank identifiers, which we will convert to categorical codes
    data_precleaning['bank_id'] = data_precleaning.index.get_level_values('id').astype('category').codes  # Convert bank IDs to categorical codes
    FEATURE_VARIABLES = ['bank_id'] + FEATURE_VARIABLES  # Add bank_id to the feature list

# Additional cleaning
nr_cells = data_precleaning.shape[0] * data_precleaning.shape[1]
nr_na = data_precleaning.isna().sum().sum()
print(f"Number of NaNs in data: {nr_na} ({nr_na / nr_cells * 100:.2f}%)")
print(f"Removing NAs from the dataset...")

# Initialize 'data' from 'data_precleaning' to work with
data = data_precleaning.copy()

data = data.dropna(subset=FEATURE_VARIABLES + [TARGET_VARIABLE]).copy()

#endregion




#region Pre-process
# Delete outliers in the target variable
from matplotlib.pylab import f


def remove_outliers(df, target_col, threshold=3):
    """Remove outliers based on z-score."""
    z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
    return df[z_scores < threshold]
print(f"Removing outliers from the target variable '{TARGET_VARIABLE}'...")
nr_cells_before = data.shape[0]
data = remove_outliers(data, TARGET_VARIABLE)
print(f"Number of observations before removing outliers: {nr_cells_before}")
nr_cells_after = data.shape[0]
print(f"Number of observations after removing outliers: {nr_cells_after}")


# Remove banks with too few observations
min_observations = 10  # Minimum number of observations per bank
def remove_banks_with_few_observations(df, min_observations=min_observations):
    """Remove banks with fewer than a specified number of observations."""
    bank_counts = df.index.get_level_values('id').value_counts()
    banks_to_remove = bank_counts[bank_counts < min_observations].index
    return df[~df.index.get_level_values('id').isin(banks_to_remove)]
print(f"Removing banks with fewer than {min_observations} observations...")
nr_cells_before = data.shape[0]
print(f"Number of observations before removing banks: {nr_cells_before}")
data = remove_banks_with_few_observations(data, min_observations=10)
nr_cells_after = data.shape[0]
print(f"Number of observations after removing banks: {nr_cells_after}")

if INCLUDE_BANK_FE:
    data['bank_id'] = 'bank_id_' + data['bank_id'].astype(str)
#endregion




#region Restrict sample

# Write a function that restricts the bank sample. There are two options: random sampling or selecting banks with average total assets in the last 5 years above a certain treshold. The threshold is 100bln by default. 
# Note: "last 5 years" is not implemented; current logic uses overall average total assets.


# Default arguments use the global variables from cell 3 (RESTRICT_TO_NUMBER_OF_BANKS, etc.)
def restrict_sample(df, 
                    begin_date = DATA_BEGIN,
                    end_date = DATA_END,
                    number_of_banks = RESTRICT_TO_NUMBER_OF_BANKS, 
                    bank_size       = RESTRICT_TO_BANK_SIZE, # Interpreted as asset threshold
                    deposit_ratio_threshold = RESTRICT_TO_MINIMAL_DEPOSIT_RATIO, 
                    max_change_deposit_ratio = RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO): # Interpreted as deposit ratio threshold
    
    df_processed = df.copy()

    # --- Sequential Filtering ---

    if begin_date is not None:
        df_processed = df_processed[df_processed.index.get_level_values('date') >= begin_date]
        print(f"Filtered to {df_processed.index.get_level_values('date').min()} and later dates.")

    if end_date is not None:
        df_processed = df_processed[df_processed.index.get_level_values('date') <= end_date]
        print(f"Filtered to {df_processed.index.get_level_values('date').max()} and earlier dates.")
    if df_processed.empty:
        print("Warning: DataFrame is empty after date filtering. Returning empty DataFrame.")
        return df_processed.iloc[0:0]

    if max_change_deposit_ratio is not None:
        df_processed['abs_change'] = df_processed.groupby(level='id')['deposit_ratio'].diff().abs().fillna(0)
        df_processed = df_processed[df_processed['abs_change'] <= max_change_deposit_ratio]
        
        if df_processed.empty and not df.empty and number_of_banks is None: # Check if this filter (when it's the first active one) emptied DataFrame
             print("Warning: DataFrame became empty after max change in deposit ratio filter.")
             return df_processed

    # 1. Filter by bank_size (asset threshold)
    if bank_size is not None: # bank_size is the threshold for average total assets
        if df_processed.empty:
            print("DataFrame is already empty before bank_size filter. Skipping.")
        elif 'total_assets' not in df_processed.columns:
            print("Warning: 'total_assets' column not found. Cannot filter by bank_size.")
        else:
            avg_assets = df_processed.groupby(level='id')['total_assets'].mean()
            ids_meeting_size_criteria = avg_assets[avg_assets >= bank_size].index
            
            if not ids_meeting_size_criteria.empty:
                df_processed = df_processed[df_processed.index.get_level_values('id').isin(ids_meeting_size_criteria)]
                print(f"Filtered to {df_processed.index.get_level_values('id').nunique()} banks with average total assets >= {bank_size}.")
            else:
                print(f"Warning: No banks in current df_processed meet average total assets >= {bank_size}. DataFrame will be empty.")
                return df_processed.iloc[0:0] # Return empty DataFrame with original columns
        
        if df_processed.empty and not df.empty and number_of_banks is None: # Check if this filter (when it's the first active one) emptied DataFrame
             print("Warning: DataFrame became empty after bank_size filter.")
             return df_processed

    # 2. Filter by deposit_ratio_threshold (minimum deposit ratio for a bank)
    if deposit_ratio_threshold is not None:
        if df_processed.empty:
            print("DataFrame is already empty before deposit_ratio_threshold filter. Skipping.")
        elif 'deposit_ratio' not in df_processed.columns:
            print("Warning: 'deposit_ratio' column not found. Cannot filter by deposit_ratio_threshold.")
        else:
            min_ratios_per_bank = df_processed.groupby(level='id')['deposit_ratio'].min()
            ids_meeting_deposit_criteria = min_ratios_per_bank[min_ratios_per_bank >= deposit_ratio_threshold].index

            if not ids_meeting_deposit_criteria.empty:
                df_processed = df_processed[df_processed.index.get_level_values('id').isin(ids_meeting_deposit_criteria)]
                print(f"Filtered to {df_processed.index.get_level_values('id').nunique()} banks with minimum deposit ratio >= {deposit_ratio_threshold}.")
            else:
                print(f"Warning: No banks in current df_processed meet minimum deposit ratio >= {deposit_ratio_threshold}. DataFrame will be empty.")
                return df_processed.iloc[0:0]
        
        if df_processed.empty and not df.empty and number_of_banks is None and bank_size is None: # Check if this filter (when it's the first active one) emptied DataFrame
             print("Warning: DataFrame became empty after deposit_ratio_threshold filter.")
             return df_processed
        
        # 3. Filter by number_of_banks (random sampling)
    if number_of_banks is not None:
        if not isinstance(number_of_banks, int) or number_of_banks <= 0:
            print(f"Warning: 'number_of_banks' must be a positive integer. Value: {number_of_banks}. Skipping this filter.")
        else:
            current_bank_ids_in_df = df_processed.index.get_level_values('id').unique()
            if len(current_bank_ids_in_df) == 0:
                print(f"No banks available for sampling from df_processed (input to this filter step is empty or has no banks).")
            elif len(current_bank_ids_in_df) > number_of_banks:
                sampled_ids = pd.Series(current_bank_ids_in_df).sample(n=number_of_banks, random_state=42, replace=False).values
                df_processed = df_processed[df_processed.index.get_level_values('id').isin(sampled_ids)]
                print(f"Randomly selected {df_processed.index.get_level_values('id').nunique()} banks.")
            else:
                print(f"Requested {number_of_banks} banks for sampling, but only {len(current_bank_ids_in_df)} available in df_processed. Using all available from current set.")
        
        if df_processed.empty and not df.empty: # Check if this filter step emptied a non-empty DataFrame
            print("Warning: DataFrame became empty after number_of_banks filter.")
            return df_processed # Early exit if empty
                
    return df_processed

# Restrict the bank sample if configured
print(f"Restrict bank sample in 'data'.")
data_before_restriction_shape = data.shape
nr_banks_before_restriction = data.index.get_level_values('id').nunique()

data = restrict_sample(data, 
                            bank_size=RESTRICT_TO_BANK_SIZE, 
                            number_of_banks=RESTRICT_TO_NUMBER_OF_BANKS, 
                            deposit_ratio_threshold=RESTRICT_TO_MINIMAL_DEPOSIT_RATIO 
                            )

print(f"Number of banks before restriction: {nr_banks_before_restriction}, after restriction: {data.index.get_level_values('id').nunique() if not data.empty else 0}")
print(f"Shape of 'data' before restriction: {data_before_restriction_shape}, after restriction: {data.shape}")
if data.empty and data_before_restriction_shape[0] > 0 :
    print("Warning: 'data' DataFrame is empty after bank sample restriction. Check threshold or sampling logic.")

X = data[FEATURE_VARIABLES].copy()
y = data[TARGET_VARIABLE].copy()

#endregion





#region Run regressions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
import sklearn.dummy
from xgboost import XGBRegressor
from prophet import Prophet # Make sure Prophet is installed
import joblib # For saving/loading models
import os # For creating directories
from scipy.stats import uniform, randint # For RandomizedSearchCV distributions
from IPython.display import display # For better display in Jupyter


def prepare_data_for_horizon(
    df_full: pd.DataFrame,
    target_col_name: str,
    features_list: list,
    horizon: int,
    train_test_split_dimension: str,
    test_ratio: float
) -> tuple:
    """
    Prepares features (X) and target (y) for a given forecast horizon from panel data.
    Handles MultiIndex, creates multi-step target, splits data, and scales features.
    """
    if not isinstance(df_full.index, pd.MultiIndex) or \
       not all(name in df_full.index.names for name in ['id', 'date']):
        if 'id' in df_full.columns and 'date' in df_full.columns:
            print("Input DataFrame does not have ('id', 'date') MultiIndex. Setting it now.")
            df_full = df_full.set_index(['id', 'date'])
        else:
            raise ValueError("df_full must have a MultiIndex with 'id' and 'date' levels, "
                             "or 'id' and 'date' columns to set as MultiIndex.")
    df_full = df_full.sort_index() # Ensure data is sorted by id and then date

    df_h = df_full.copy() # Work on a copy
    shifted_target_col = f'{target_col_name}_target_h{horizon}'
    # Group by 'id' to ensure the shift is done within each unit's timeline
    df_h[shifted_target_col] = df_h.groupby(level='id', group_keys=False)[target_col_name].shift(-horizon)
    df_h.dropna(subset=[shifted_target_col], inplace=True) # Drop NaNs created by shifting

    if df_h.empty:
        print(f"DataFrame is empty for horizon {horizon} after creating target. Skipping.")
        return None, None, None, None, None, None
    
    # --- Preliminary Split (before lag creation for main models) ---
    # We will split df_h (which has the target) into preliminary train/test.
    # Then create lags separately or in a way that respects this split.

    y_full = df_h[shifted_target_col]
    X_pre_lag_full = df_h[features_list].copy() # Contemporaneous features for now

    if X_pre_lag_full.empty or y_full.empty:
        print(f"X or y is empty for horizon {horizon} before splitting. Skipping.")
        return None, None, None, None, None, None

    X_train_orig_df, X_test_orig_df, y_train, y_test = None, None, None, None

    if train_test_split_dimension == "date":
        # Use X_for_main_models for splitting decisions, as it's the primary dataset for most models
        unique_dates = X_pre_lag_full.index.get_level_values('date').unique().sort_values()
        if len(unique_dates) < 2: # Need at least two unique dates to split
            print(f"Not enough unique dates to perform a chronological split for horizon {horizon}. Skipping.")
            return None, None, None, None, None, None

        split_idx = int(len(unique_dates) * (1 - test_ratio))
        if split_idx == 0 or split_idx >= len(unique_dates): # Avoid empty train or test set by date
            print(f"Test ratio results in empty train or test set based on unique dates for horizon {horizon}. Skipping.")
            return None, None, None, None, None, None

        split_date = unique_dates[split_idx -1] # Data UP TO this date for train

        train_mask = X_pre_lag_full.index.get_level_values('date') <= split_date
        test_mask = X_pre_lag_full.index.get_level_values('date') > split_date

        X_train_pre_lag = X_pre_lag_full[train_mask]
        X_test_pre_lag = X_pre_lag_full[test_mask]
        y_train = y_full[train_mask]
        y_test = y_full[test_mask]

    elif train_test_split_dimension == "id":
        # Use X_for_main_models for splitting decisions
        unique_ids_in_X = X_pre_lag_full.index.get_level_values('id').unique()
        if len(unique_ids_in_X) < 2:
            print(f"Not enough unique IDs ({len(unique_ids_in_X)}) for split by ID (horizon {horizon}). Skipping.")
            return None, None, None, None, None, None

        # Calculate number of test IDs, ensure it's at least 1 if ratio > 0 and less than total
        num_test_ids = int(np.ceil(len(unique_ids_in_X) * test_ratio))
        if test_ratio > 0 and num_test_ids == 0: num_test_ids = 1
        if num_test_ids == 0 or num_test_ids >= len(unique_ids_in_X):
            print(f"Test ratio ({test_ratio}) for {len(unique_ids_in_X)} IDs results in invalid test set size ({num_test_ids}). Skipping for horizon {horizon}.")
            return None, None, None, None, None, None

        X_pre_lag_reset = X_pre_lag_full.reset_index()
        y_full_reset = y_full.reset_index()

        gss = GroupShuffleSplit(n_splits=1, test_size=num_test_ids, random_state=42)
        try:
            train_idx, test_idx = next(gss.split(X_pre_lag_reset, y_full_reset, groups=X_pre_lag_reset['id']))
        except ValueError as e:
            print(f"Error during GroupShuffleSplit for horizon {horizon}: {e}. Skipping.")
            return None, None, None, None, None, None

        X_train_pre_lag = X_pre_lag_reset.iloc[train_idx].set_index(['id', 'date']).sort_index()
        X_test_pre_lag = X_pre_lag_reset.iloc[test_idx].set_index(['id', 'date']).sort_index()
        y_train = y_full_reset.iloc[train_idx].set_index(['id', 'date'])[shifted_target_col].sort_index()
        y_test = y_full_reset.iloc[test_idx].set_index(['id', 'date'])[shifted_target_col].sort_index()
    else:
        raise ValueError("train_test_split_dimension must be 'date' or 'id'")

    # --- Feature Engineering: Lags (Applied after initial train/test split) ---
    base_numeric_features = [f for f in features_list if f in X_train_pre_lag.columns and pd.api.types.is_numeric_dtype(X_train_pre_lag[f]) and f != target_col_name]
    base_categorical_features = [f for f in features_list if f in X_train_pre_lag.columns and f not in base_numeric_features and f != target_col_name]

    def _add_lags_to_df(df_to_lag, numeric_features_to_lag, num_lags_to_add):
        df_out = df_to_lag.copy()
        lagged_feature_names = []
        if num_lags_to_add > 0:
            for feature_name in numeric_features_to_lag:
                for lag_num in range(1, num_lags_to_add + 1):
                    lagged_col_name = f"{feature_name}_lag_{lag_num}"
                    df_out[lagged_col_name] = df_out.groupby(level='id', group_keys=False)[feature_name].shift(lag_num)
                    lagged_feature_names.append(lagged_col_name)
        return df_out, lagged_feature_names

    # Create lags for training data (using only training data history)
    X_train_with_lags_df, train_lagged_feature_names = _add_lags_to_df(X_train_pre_lag, base_numeric_features, NUMBER_OF_LAGS_TO_INCLUDE)
    
    # Create lags for test data
    # To do this correctly, test data can use contemporaneous values from train data if lags span across the split point.
    # Concatenate, create lags, then split again.
    if not X_test_pre_lag.empty:
        combined_for_test_lags = pd.concat([X_train_pre_lag, X_test_pre_lag]) # Ensure correct order if not already sorted
        combined_for_test_lags = combined_for_test_lags.sort_index() # Sort by id, then date
        
        X_combined_with_lags_df, test_lagged_feature_names = _add_lags_to_df(combined_for_test_lags, base_numeric_features, NUMBER_OF_LAGS_TO_INCLUDE)
        X_test_with_lags_df = X_combined_with_lags_df.loc[X_test_pre_lag.index]
    else:
        X_test_with_lags_df = pd.DataFrame(index=X_test_pre_lag.index) # Empty df with same index
        test_lagged_feature_names = [] # Should be same as train_lagged_feature_names if NUMBER_OF_LAGS_TO_INCLUDE is same

    # Define final model features (contemporaneous categorical + lagged numeric OR contemporaneous numeric)
    actual_model_features = base_categorical_features.copy()
    if NUMBER_OF_LAGS_TO_INCLUDE > 0:
        actual_model_features.extend(train_lagged_feature_names) # Use names from train lag creation
    else:
        actual_model_features.extend(base_numeric_features)

    # Select final features and drop NaNs from lagging
    X_train_main_unscaled = X_train_with_lags_df[actual_model_features].dropna()
    y_train = y_train.loc[X_train_main_unscaled.index] # Align y_train with X_train after dropna

    if not X_test_with_lags_df.empty:
        X_test_main_unscaled = X_test_with_lags_df[actual_model_features].dropna() # Use same feature list
        y_test = y_test.loc[X_test_main_unscaled.index] # Align y_test
    else:
        X_test_main_unscaled = pd.DataFrame(columns=actual_model_features, index=X_test_with_lags_df.index) # Empty df with correct columns

    # Data for Prophet (uses original contemporaneous features from the split data)
    # Align X_train_prophet_unscaled with the final y_train indices (after lagging and dropna on main X_train)
    X_train_prophet_unscaled = X_train_pre_lag.loc[y_train.index] 
    if not X_test_main_unscaled.empty :
        X_test_prophet_unscaled = X_test_pre_lag.loc[y_test.index]
    else: # If X_test_main_unscaled became empty (e.g. all rows had NaNs after lagging)
        X_test_prophet_unscaled = pd.DataFrame(columns=features_list, index=X_test_pre_lag.index)

    if X_train_main_unscaled.empty or y_train.empty : # Check after all processing for train
        print(f"Train or test set is empty after split for horizon {horizon}. Check test_ratio and data. Skipping.")
        return None, None, None, None, None, None

    # --- Feature Transformation (Scaling for numeric, OHE for categorical) ---
    # Apply to X_train_main_unscaled, which contains the features for the main models (potentially lagged)
    numeric_cols_to_scale = X_train_main_unscaled.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_to_ohe = X_train_main_unscaled.select_dtypes(include=['object', 'category']).columns.tolist()

    transformers = []
    if numeric_cols_to_scale:
        transformers.append(('num', StandardScaler(), numeric_cols_to_scale))
    if categorical_cols_to_ohe:
        # drop=None to handle cases where a category might appear in test but not train if handle_unknown='ignore'
        # or if a single category is present in a fold. 'if_binary' is safer for single unique values after split.
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None), categorical_cols_to_ohe))

    if not transformers:
        print(f"Warning: No features to scale or OHE for horizon {horizon} from actual_model_features. Using unscaled/unencoded features (potential error downstream).")
        # This path might lead to errors if non-numeric data remains.
        # Ensure actual_model_features are correctly identified.
        X_train_final = X_train_main_unscaled.copy()
        X_test_final = X_test_main_unscaled.copy()
    else:
        # All columns in features_list should ideally be covered by numeric_cols_to_scale or categorical_cols_to_ohe.
        # If not, 'remainder="drop"' will remove them. If they are needed, 'remainder="passthrough"' would be an option,
        # but then they must be model-compatible.
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        X_train_transformed_np = preprocessor.fit_transform(X_train_main_unscaled)
        
        try:
            transformed_feature_names = preprocessor.get_feature_names_out()
        except AttributeError: # Fallback for older scikit-learn versions
            transformed_feature_names = []
            for name_trans, trans_obj, columns_trans_list in preprocessor.transformers_:
                if trans_obj == 'drop' or trans_obj == 'passthrough': # Should not be 'drop' or 'passthrough' here
                    continue
                if hasattr(trans_obj, 'get_feature_names_out'): # e.g. OneHotEncoder
                    transformed_feature_names.extend(trans_obj.get_feature_names_out(columns_trans_list))
                else: # e.g. StandardScaler, which doesn't change names but returns a list of columns it processed
                    if isinstance(columns_trans_list, str): # Should be list from ColumnTransformer
                         transformed_feature_names.append(columns_trans_list)
                    else:
                         transformed_feature_names.extend(columns_trans_list)
        
        X_train_final = pd.DataFrame(X_train_transformed_np, columns=transformed_feature_names, index=X_train_main_unscaled.index)

        if not X_test_main_unscaled.empty:
            # Ensure X_test_main_unscaled has the same columns as X_train_main_unscaled before transform
            # This relies on X_test_main_unscaled having all columns that preprocessor was fit on.
            X_test_transformed_np = preprocessor.transform(X_test_main_unscaled)
            X_test_final = pd.DataFrame(X_test_transformed_np, columns=transformed_feature_names, index=X_test_main_unscaled.index)
        else:
            X_test_final = pd.DataFrame(columns=transformed_feature_names, index=X_test_main_unscaled.index)

    # X_train_final and X_test_final are now the fully preprocessed numeric dataframes
    return X_train_final, X_test_final, y_train, y_test, X_train_prophet_unscaled, X_test_prophet_unscaled # Return aligned prophet data

def train_evaluate_model(model_name_key, model_instance, X_train, y_train, X_test, y_test, 
                         param_grid=None, cv_splitter=None, 
                         use_random_search=False, n_iter_random_search=10): # Added random search params
    """Trains a single model, tunes if param_grid is provided, and evaluates it."""
    best_model = model_instance # Start with the original instance

    is_xgb = isinstance(model_instance, XGBRegressor)
    fit_params = {}

    # Prepare eval_set for XGBoost if early stopping is active
    # Ensure X_test and y_test are not empty and have consistent lengths
    if is_xgb and hasattr(model_instance, 'early_stopping_rounds') and model_instance.early_stopping_rounds is not None:
        if not X_test.empty and not y_test.empty and len(X_test) == len(y_test):
            # Create a temporary evaluation set from the end of the training data if X_test is very small or for robustness
            # This is a common practice if the provided X_test for final eval shouldn't be seen during tuning.
            # However, for early stopping during the *final* fit, X_test is fine.
            # For GridSearchCV, a separate validation set from train split is used internally.
            # Here, this eval_set is for the *final* model fit after potential tuning.
            fit_params = {'eval_set': [(X_test, y_test)], 'verbose': False}
        else:
            print(f"    XGBoost: Skipping eval_set for early stopping due to empty or misaligned X_test/y_test for {model_name_key}")


    if param_grid and cv_splitter and not X_train.empty:
        print(f"    Tuning {model_name_key}...")
        # Check for sufficient samples for CV
        min_samples_for_cv = cv_splitter.get_n_splits() * 2 # Rough estimate, actual min depends on TimeSeriesSplit logic
        if len(X_train) < min_samples_for_cv :
             print(f"    Skipping tuning for {model_name_key}: not enough samples ({len(X_train)}) for {cv_splitter.get_n_splits()} CV splits. Fitting directly.")
             # Fit directly
             if is_xgb and fit_params: best_model.fit(X_train, y_train, **fit_params)
             else: best_model.fit(X_train, y_train)

        # Check for constant target variable which can cause issues with some models/solvers in CV
        elif len(np.unique(y_train)) == 1 and model_name_key not in ["DummyRegressor", "DecisionTree", "XGBoost"]: # Some models are fine with constant y
            print(f"    Skipping tuning for {model_name_key}: target variable is constant. Fitting directly.")
            best_model.fit(X_train, y_train)
        else:
            try:
                if use_random_search:
                    search = RandomizedSearchCV(model_instance, param_distributions=param_grid, 
                                                n_iter=n_iter_random_search, cv=cv_splitter, 
                                                scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
                else:
                    search = GridSearchCV(model_instance, param_grid, cv=cv_splitter, 
                                          scoring='neg_mean_squared_error', n_jobs=-1)
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                print(f"    Best params for {model_name_key}: {search.best_params_}")
            except Exception as e:
                print(f"    Error during GridSearchCV for {model_name_key}: {e}. Fitting with default params.")
                if is_xgb and fit_params: best_model.fit(X_train, y_train, **fit_params)
                else: best_model.fit(X_train, y_train)
    elif not X_train.empty: # Fit directly if no tuning or if X_train is not empty
        print(f"    Fitting {model_name_key} with default parameters (no tuning).")
        if is_xgb and fit_params: best_model.fit(X_train, y_train, **fit_params)
        else: best_model.fit(X_train, y_train)
    else: # X_train is empty
        print(f"    Skipping fitting for {model_name_key} as X_train is empty.")
        # Return NaNs but keep a model instance if needed (though it's unfitted)
        return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': model_instance}


    if X_test.empty or y_test.empty:
        print(f"    Skipping prediction for {model_name_key} as X_test or y_test is empty.")
        return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'model_object': best_model}

    try:
        predictions_test = best_model.predict(X_test)
        if not X_train.empty and len(X_train) > 0: # Ensure X_train is not empty
            predictions_train = best_model.predict(X_train)
        else:
            predictions_train = np.array([]) # Empty array if no train data to predict on
    except Exception as e:
        print(f"    Error during prediction for {model_name_key}: {e}")
        return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': best_model}


    # Calculate test metrics
    mae = mean_absolute_error(y_test, predictions_test)
    mse = mean_squared_error(y_test, predictions_test)
    rmse_test = np.sqrt(mse)
    r2 = r2_score(y_test, predictions_test)
    
    # MAPE calculation with check for zeros in y_test
    y_test_no_zeros = y_test.replace(0, 0.00001).dropna() # Replace 0 with a small number to avoid division by zero
    if not y_test_no_zeros.empty:
        predictions_no_zeros = pd.Series(predictions_test, index=y_test.index).loc[y_test_no_zeros.index]
        mape = np.mean(np.abs((y_test_no_zeros - predictions_no_zeros) / y_test_no_zeros)) * 100
    else: # Handle case where y_test_no_zeros might be empty (e.g. all y_test were 0 or NaN)
        mape = np.nan
        
    # Calculate train RMSE
    rmse_train = np.nan
    if len(predictions_train) > 0 and len(predictions_train) == len(y_train) and not y_train.empty:
        try:
            rmse_train = np.sqrt(mean_squared_error(y_train, predictions_train))
        except Exception as e_train_rmse:
            print(f"    Could not calculate train RMSE for {model_name_key}: {e_train_rmse}")
            rmse_train = np.nan
    elif not y_train.empty and len(predictions_train) == 0 : # If train data exists but no predictions were made (e.g. X_train was empty)
        print(f"    Skipping train RMSE for {model_name_key} as no train predictions were made (X_train might have been empty).")
    elif not y_train.empty: # General mismatch
        print(f"    Skipping train RMSE for {model_name_key} due to prediction/data mismatch (preds: {len(predictions_train)}, y_train: {len(y_train)}).")

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse_test, 'R2': r2, 'MAPE': mape, 'RMSE_train': rmse_train, 'model_object': best_model}

def train_evaluate_prophet(X_train_orig, y_train, X_test_orig, y_test, feature_variables_list):
    """Trains and evaluates a Prophet model, using 'date' for 'ds'."""
    print("    Training Prophet model...")
    if X_train_orig.empty or y_train.empty:
        print("    Prophet: Skipping due to empty training input data (X_train_orig or y_train).")
        return None # Or return dict with NaNs
    if X_test_orig.empty or y_test.empty: # If test data is empty, can't evaluate
        print("    Prophet: Skipping evaluation due to empty test input data (X_test_orig or y_test).")
        # Fit a model but return NaN metrics
        df_train_prophet_fit_only = X_train_orig.reset_index()
        df_train_prophet_fit_only['ds'] = pd.to_datetime(df_train_prophet_fit_only['date'])
        df_train_prophet_fit_only['y'] = y_train.values
        m_prophet_fit_only = Prophet()
        valid_regressors_fit_only = [reg for reg in feature_variables_list if reg in df_train_prophet_fit_only.columns]
        for reg in valid_regressors_fit_only: m_prophet_fit_only.add_regressor(reg)
        try:
            m_prophet_fit_only.fit(df_train_prophet_fit_only[['ds', 'y'] + valid_regressors_fit_only])
        except Exception as e_fit:
            print(f"    Error during Prophet fitting (no test eval): {e_fit}")
            m_prophet_fit_only = None # Model fitting failed
        return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': m_prophet_fit_only}


    # Prepare training data for Prophet
    df_train_prophet = X_train_orig.reset_index() # Get 'id' and 'date' as columns
    df_train_prophet['ds'] = pd.to_datetime(df_train_prophet['date']) # Ensure 'date' is datetime
    df_train_prophet['y'] = y_train.values # y_train is a Series, align its values

    # Prepare future dataframe for Prophet
    df_future_prophet = X_test_orig.reset_index()
    df_future_prophet['ds'] = pd.to_datetime(df_future_prophet['date'])

    # Ensure regressor columns exist
    valid_regressors = [reg for reg in feature_variables_list if reg in df_train_prophet.columns and reg in df_future_prophet.columns]
    if len(valid_regressors) != len(feature_variables_list):
        missing = set(feature_variables_list) - set(valid_regressors)
        print(f"    Prophet: Regressors missing from data: {missing}. Using available: {valid_regressors}")
    
    m_prophet = Prophet() # Add seasonality/changepoint args if needed
    for regressor in valid_regressors:
        m_prophet.add_regressor(regressor)

    try:
        m_prophet.fit(df_train_prophet[['ds', 'y'] + valid_regressors])
        
        # Predict on test dataframe
        # Prophet requires 'ds' and all regressor columns in the future_df
        forecast_prophet_test = m_prophet.predict(df_future_prophet[['ds'] + valid_regressors])
        
        # Align predictions with y_test.
        # y_test has MultiIndex. Prophet's forecast_prophet_test does not.
        # This alignment assumes the order of X_test_orig (and thus df_future_prophet)
        # matches the order of y_test when their values are taken.
        # And that Prophet returns predictions for every row in df_future_prophet.
        if len(forecast_prophet_test['yhat']) != len(y_test):
            print(f"    Prophet WARNING: Test forecast length ({len(forecast_prophet_test['yhat'])}) mismatch with y_test ({len(y_test)}). Slicing predictions.")
            # This could indicate issues if Prophet drops dates or if there's misalignment.
            # A more robust alignment might involve merging on 'ds' and original index if possible.
        predictions_prophet_test = forecast_prophet_test['yhat'].values[:len(y_test)]

        # Predict on train dataframe
        rmse_train_prophet = np.nan
        if not df_train_prophet.empty and len(df_train_prophet) > 0:
            forecast_prophet_train = m_prophet.predict(df_train_prophet[['ds'] + valid_regressors])
            if len(forecast_prophet_train['yhat']) == len(y_train):
                predictions_prophet_train = forecast_prophet_train['yhat'].values
                rmse_train_prophet = np.sqrt(mean_squared_error(y_train, predictions_prophet_train))
            else:
                print(f"    Prophet WARNING: Train forecast length ({len(forecast_prophet_train['yhat'])}) mismatch with y_train ({len(y_train)}).")

        # Calculate test metrics
        mae = mean_absolute_error(y_test, predictions_prophet_test)
        mse = mean_squared_error(y_test, predictions_prophet_test)
        rmse_test = np.sqrt(mse)
        r2 = r2_score(y_test, predictions_prophet_test)

        y_test_no_zeros = y_test.replace(0, 0.00001).dropna()
        if not y_test_no_zeros.empty:
            predictions_no_zeros = pd.Series(predictions_prophet_test, index=y_test.index).loc[y_test_no_zeros.index]
            mape = np.mean(np.abs((y_test_no_zeros - predictions_no_zeros) / y_test_no_zeros)) * 100
        else:
            mape = np.nan
            
        return {
            'MAE': mae, 'MSE': mse, 'RMSE': rmse_test, 'R2': r2, 'MAPE': mape, 
            'RMSE_train': rmse_train_prophet, 'model_object': m_prophet
        }
    except Exception as e:
        print(f"    Error during Prophet training/prediction: {e}")
        return None # Or dict with NaNs and no model object

def train_evaluate_ensemble(trained_models_dict, X_train_scaled, y_train, X_test_scaled, y_test):
    """Trains and evaluates a VotingRegressor ensemble model."""
    print("    Attempting to train Voting Regressor ensemble...")
    if X_train_scaled.empty or y_train.empty:
        print("    VotingEnsemble: Skipping due to empty training data.")
        return None
    if X_test_scaled.empty or y_test.empty:
        print("    VotingEnsemble: Skipping evaluation due to empty test data.")
        return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None} # No model if no test data

    estimators = []
    # Example: Use specific, well-performing, diverse models
    # Ensure model objects exist and are not None (e.g. if a base model failed)
    potential_base_models = ["XGBoost", "Ridge", "ElasticNet"] # Prioritize robust models
    for model_name_key in potential_base_models:
        if model_name_key in trained_models_dict and \
           trained_models_dict[model_name_key] and \
           trained_models_dict[model_name_key].get('model_object') is not None:
            # Check if the model object is actually a fitted estimator
            base_model_obj = trained_models_dict[model_name_key]['model_object']
            try: # Check if model is fitted (specific checks might be needed per model type)
                if hasattr(base_model_obj, "coef_") or hasattr(base_model_obj, "feature_importances_") or \
                   (isinstance(base_model_obj, XGBRegressor) and base_model_obj._Booster is not None): # Basic checks
                    estimators.append((model_name_key.lower(), base_model_obj))
                else:
                    print(f"    Base model {model_name_key} for ensemble seems not fitted. Skipping.")
            except AttributeError: # Some models might not have these, or not be sklearn estimators
                 print(f"    Could not verify if base model {model_name_key} is fitted. Skipping for ensemble.")


    if len(estimators) < 2: # VotingRegressor needs at least two estimators
        print("    Not enough suitable (fitted) base models for Voting Regressor. Skipping ensemble.")
        return None

    try:
        voting_reg = VotingRegressor(estimators=estimators, n_jobs=-1)
        # Fit the ensemble - it will re-fit the base estimators if they were not cloned and pre-fitted.
        # The model_objects from results_store are already fitted.
        voting_reg.fit(X_train_scaled, y_train)
        
        predictions_test = voting_reg.predict(X_test_scaled)
        rmse_train_ensemble = np.nan
        if not X_train_scaled.empty and len(X_train_scaled) > 0:
            predictions_train = voting_reg.predict(X_train_scaled)
            if len(predictions_train) == len(y_train):
                 rmse_train_ensemble = np.sqrt(mean_squared_error(y_train, predictions_train))
            else:
                print(f"    VotingEnsemble WARNING: Train forecast length ({len(predictions_train)}) mismatch with y_train ({len(y_train)}).")

        mae = mean_absolute_error(y_test, predictions_test)
        mse = mean_squared_error(y_test, predictions_test)
        rmse_test = np.sqrt(mse)
        r2 = r2_score(y_test, predictions_test)
        y_test_no_zeros = y_test.replace(0, 0.00001).dropna()
        if not y_test_no_zeros.empty:
            predictions_no_zeros = pd.Series(predictions_test, index=y_test.index).loc[y_test_no_zeros.index]
            mape = np.mean(np.abs((y_test_no_zeros - predictions_no_zeros) / y_test_no_zeros)) * 100
        else:
            mape = np.nan
        
        print("    Voting Regressor trained successfully.")
        return {
            'MAE': mae, 
            'MSE': mse, 
            'RMSE': rmse_test, 
            'R2': r2, 
            'MAPE': mape, 
            'RMSE_train': rmse_train_ensemble, 
            'model_object': voting_reg}
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
            # Ensure metrics is a dictionary before trying to get keys
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
            else: # Handle cases where metrics might be None or not a dict
                 summary_entries.append({
                    'Horizon': horizon,
                    'Model': model_name, # Still record the model attempt
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
        return pd.DataFrame() # Return empty DataFrame if no results

    # --- Display General Summary ---
    summary_df = pd.DataFrame(summary_entries)
    summary_df = summary_df.sort_values(by=['Horizon', 'RMSE'])

    if print_final_summary:
        print("\nOverall Performance Metrics (Test Set):")
        display(summary_df) # Use IPython display for better notebook formatting

    if plot_result_charts:
        if summary_df.empty or not FORECAST_HORIZONS or summary_df['RMSE'].isnull().all():
            print("Cannot plot results: summary is empty, RMSE is all NaN, or FORECAST_HORIZONS not defined.")
            return summary_df
            
        plt.figure(figsize=(14, 8))
        unique_models = summary_df['Model'].unique()
        for model_name_plot in unique_models: # Changed variable name to avoid conflict
            model_data = summary_df[summary_df['Model'] == model_name_plot].dropna(subset=['RMSE', 'Horizon']) # Ensure Horizon is also not NaN
            if not model_data.empty:
                 plt.plot(model_data['Horizon'].astype(str), model_data['RMSE'], marker='o', linestyle='-', label=model_name_plot) # Horizon as str for categorical plotting
        
        plt.xlabel("Forecast Horizon (Quarters)")
        plt.ylabel("Root Mean Squared Error (RMSE)")
        plt.title("Model RMSE vs. Forecast Horizon", fontsize=16)
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # plt.xticks(FORECAST_HORIZONS) # If horizons are numeric and you want specific ticks
        plt.tight_layout() # Adjust layout to make room for legend
        plt.show()

    # --- Display Train vs. Test RMSE Summary ---
    if train_test_rmse_entries and print_final_summary:
        train_test_rmse_df = pd.DataFrame(train_test_rmse_entries)
        train_test_rmse_df = train_test_rmse_df.sort_values(by=['Horizon', 'RMSE_Test'])
        print("\n\n--- Training vs. Test RMSE Comparison ---")
        display(train_test_rmse_df)

    return summary_df
#endregion





#region Execute

# Initialize results store

results_store = {}

if SAVE_ARTIFACTS:
    os.makedirs(ARTIFACTS_BASE_DIR, exist_ok=True)
    print(f"Artifacts will be saved in: {os.path.abspath(ARTIFACTS_BASE_DIR)}")

models_config, param_grids_config = get_models_and_param_grids(use_random_search=USE_RANDOM_SEARCH_CV, n_iter_random_search=N_ITER_RANDOM_SEARCH)


# Use the 'data' DataFrame directly, as it already contains all FEATURE_VARIABLES and the TARGET_VARIABLE
# This avoids creating duplicate columns if TARGET_VARIABLE is also in FEATURE_VARIABLES.
all_data = data.copy() # Create a copy to ensure 'data' isn't modified if 'all_data' is changed elsewhere

# --- Iterate through each forecast horizon ---
for horizon_val in FORECAST_HORIZONS: # Changed variable name
    print(f"\n--- Processing Horizon: {horizon_val}-quarter(s) ahead ---")
    results_store[horizon_val] = {} # Initialize results for this horizon
    
    current_horizon_artifact_dir = ""
    if SAVE_ARTIFACTS:
        current_horizon_artifact_dir = os.path.join(ARTIFACTS_BASE_DIR, f"horizon_{horizon_val}")
        os.makedirs(current_horizon_artifact_dir, exist_ok=True)

    # Prepare data for the current horizon
    prepared_data = prepare_data_for_horizon(
        all_data, TARGET_VARIABLE, FEATURE_VARIABLES, horizon_val, TRAIN_TEST_SPLIT_DIMENSION, TEST_SPLIT_RATIO
    )

    # If data preparation fails for a horizon, fill with NaNs and continue
    if prepared_data is None:
        print(f"  Skipping horizon {horizon_val} due to data preparation failure.")
        # Store NaN results for all models for this horizon to keep structure
        for name_model in models_config.keys():
            results_store[horizon_val][name_model] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
        # Also for Prophet and Ensemble if they are treated separately in results
        results_store[horizon_val]['Prophet'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
        results_store[horizon_val]['VotingEnsemble'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
        continue # Move to the next horizon
            
    X_train_scaled_df, X_test_scaled_df, y_train, y_test, X_train_orig, X_test_orig = prepared_data
    
    # TimeSeriesSplit for cross-validation (if enough data)
    current_n_splits_cv = N_SPLITS_CV
    # TimeSeriesSplit needs at least n_splits + 1 samples for the first training set to be non-empty.
    # More practically, len(X_train) should be significantly larger.
    if len(X_train_scaled_df) < current_n_splits_cv + 1 : # Stricter check for TimeSeriesSplit
        print(f"  Warning: Training set size ({len(X_train_scaled_df)}) is too small for {current_n_splits_cv} CV splits. Reducing or skipping CV.")
        if len(X_train_scaled_df) >= 3 and current_n_splits_cv > 1: # Min for 2 splits might be 3 samples.
            current_n_splits_cv = max(1, int(len(X_train_scaled_df) / 2) -1) # Heuristic reduction
            if current_n_splits_cv < 2: current_n_splits_cv = 0 # If too small, no CV
            print(f"  Reduced CV splits to {current_n_splits_cv}.")
        else:
            print(f"  Skipping CV tuning for horizon {horizon_val} due to insufficient training data.")
            current_n_splits_cv = 0 # No CV
    
    tscv_splitter = TimeSeriesSplit(n_splits=current_n_splits_cv) if current_n_splits_cv >= 2 else None # tscv needs at least 2 splits


    # --- Train and Evaluate Standard Models & Fixed Effects Model ---
    for model_name_loop, model_instance_loop in models_config.items(): # Changed variable names
        print(f"  Training {model_name_loop} for horizon {horizon_val}...")
        model_results = None # Initialize
        
        # Check if data is available before attempting to train
        if X_train_scaled_df.empty or y_train.empty:
            # Test data can be empty if test_ratio is very small or data is limited.
            # Prediction/evaluation will handle empty X_test/y_test.
            print(f"    Skipping {model_name_loop} due to empty scaled training data for horizon {horizon_val}.")
            results_store[horizon_val][model_name_loop] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
            continue

        model_results = train_evaluate_model(
            model_name_loop, model_instance_loop, X_train_scaled_df, y_train, X_test_scaled_df, y_test,
            param_grid=param_grids_config.get(model_name_loop), # Get specific param grid
            cv_splitter=tscv_splitter if param_grids_config.get(model_name_loop) and tscv_splitter else None,
            use_random_search=USE_RANDOM_SEARCH_CV, n_iter_random_search=N_ITER_RANDOM_SEARCH
        )
        
        # Store results, ensuring an entry even if model_results is None
        if model_results:
            results_store[horizon_val][model_name_loop] = model_results
        else:
            results_store[horizon_val][model_name_loop] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}

    # --- Train and Evaluate Prophet Model ---
    if X_train_orig.empty or y_train.empty : # Check before calling prophet if train data is available
         print(f"  Skipping Prophet for horizon {horizon_val} due to empty original training data.")
         results_store[horizon_val]['Prophet'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
    else:
        prophet_results = train_evaluate_prophet(X_train_orig, y_train, X_test_orig, y_test, FEATURE_VARIABLES)
        if prophet_results:
            results_store[horizon_val]['Prophet'] = prophet_results
        else: # If Prophet fails or returns None
            results_store[horizon_val]['Prophet'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}

    # --- (Optional) Train and Evaluate Ensemble Model ---
    # Check if base models were trained and data is available
    if results_store[horizon_val] and not (X_train_scaled_df.empty or y_train.empty):
        ensemble_results = train_evaluate_ensemble(results_store[horizon_val], X_train_scaled_df, y_train, X_test_scaled_df, y_test)
        if ensemble_results:
            results_store[horizon_val]['VotingEnsemble'] = ensemble_results
        else: # If ensemble fails or returns None
            results_store[horizon_val]['VotingEnsemble'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}
    else: # If no base models or no training data for ensemble
        results_store[horizon_val]['VotingEnsemble'] = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'RMSE_train': np.nan, 'model_object': None}


# --- Aggregate and Display All Results ---
final_summary_df = aggregate_and_display_results(results_store, plot_result_charts=PLOT_RESULT_CHARTS, print_final_summary=PRINT_FINAL_SUMMARY)






#region Plot the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def plot_features_timeseries_flat(
    df: pd.DataFrame, 
    date_column: str = None, 
    id_column: str = None, 
    feature_columns: List[str] = None
):
    """
    Plots each specified feature of a DataFrame as a time series in a separate subplot,
    arranged in a 3-column grid. Each unique ID (from id_column) is plotted
    with a distinct color. The DataFrame is expected to be "flat", meaning
    date and ID are regular columns.

    Args:
        df (pd.DataFrame): DataFrame with date, id, and feature columns.
        date_column (str, optional): Name of the column containing date/time data.
                                     Defaults to 'Date'.
                                     This column should be sortable (e.g., datetime objects).
        id_column (str, optional): Name of the column containing ID data for hue.
                                   Defaults to 'id'.
        feature_columns (List[str], optional): List of column names for features to plot.
                                               Defaults to all columns not used as date_column or id_column.
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # --- Determine column names if not provided ---
    if date_column is None:
        date_column = 'date'
    if id_column is None:
        id_column = 'id'
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in [date_column, id_column]]

    required_cols = [date_column, id_column] + feature_columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {', '.join(missing_cols)}")


    if not feature_columns:
        print("No feature columns provided to plot.")
        return
    
    # Ensure date column is suitable for plotting (e.g., convert to datetime if it's string)
    # For simplicity, we'll assume it's already in a plottable format (datetime or numeric)
    # but in a real scenario, you might add:
    # if not pd.api.types.is_datetime64_any_dtype(df[date_column]) and \
    #    not pd.api.types.is_numeric_dtype(df[date_column]):
    #     try:
    #         df[date_column] = pd.to_datetime(df[date_column])
    #     except Exception as e:
    #         print(f"Warning: Could not convert date_column '{date_column}' to datetime: {e}")

    num_features = len(feature_columns)

    # --- Setup for plotting ---
    num_cols_grid = 3
    num_rows_grid = (num_features + num_cols_grid - 1) // num_cols_grid

    fig, axes = plt.subplots(
        num_rows_grid, num_cols_grid, 
        figsize=(5 * num_cols_grid, 4 * num_rows_grid), 
        sharex=True # Share x-axis for easier comparison
    )

    # Flatten axes array for easy iteration, handling single row/col cases
    axes_flat: np.ndarray[plt.Axes]
    if num_rows_grid == 1 and num_cols_grid == 1:
        axes_flat = np.array([axes])
    elif num_rows_grid == 1 or num_cols_grid == 1:
        axes_flat = axes if isinstance(axes, np.ndarray) else np.array([axes])
    else:
        axes_flat = axes.flatten()

    # --- Create plots ---
    legend_handles_map = {} # Using dict to ensure unique labels for the figure legend

    for i, feature_name in enumerate(feature_columns):
        ax = axes_flat[i]
        
        sns.lineplot(
            data=df,
            x=date_column,
            y=feature_name,
            hue=id_column,
            ax=ax,
            legend=True # Generate legend for this subplot to extract handles
        )
        
        # Collect unique handles and labels for the figure-level legend
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_handles_map: # Add if label is new
                legend_handles_map[label] = handle
        
        if ax.get_legend() is not None:
            ax.get_legend().remove() # Remove individual subplot legend

        ax.set_title(f'{feature_name}')
        ax.set_xlabel(date_column) # Use the provided date_column name
        ax.set_ylabel(feature_name)
        ax.tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    for j in range(num_features, len(axes_flat)):
        fig.delaxes(axes_flat[j])



    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
    plt.show()

plot_features_timeseries_flat(all_data.reset_index(), feature_columns=[TARGET_VARIABLE] + ['log_total_assets', 'deposit_ratio'])
#endregion

#region Plot estimated vs actual values
import math
import matplotlib.pyplot as plt
import pandas as pd # Ensure pandas is imported

# --- Plotting Script (Cleaner Version) ---
# ASSUMPTIONS:
# 1. `results_store` is populated from your main script.
# 2. `TARGET_VARIABLE` and `FEATURE_VARIABLES` are defined.
# 3. `X_train_orig`, `X_test_orig`, `y_train`, `y_test`, `X_train_scaled_df`, `X_test_scaled_df`
#    are correctly set to the data corresponding to HORIZON 1 from your main script.
# 4. The `preprocess_for_fixed_effects` function is defined in a previous cell or imported.

results_h1 = results_store.get(1, {}) # Get results for horizon 1

models_to_plot = []
for model_name, metrics in results_h1.items():
    if 'model_object' in metrics and metrics['model_object'] is not None and hasattr(metrics['model_object'], 'predict'):
        # Exclude Prophet for now as it needs special handling for prediction in plots
        if model_name != "Prophet":
            models_to_plot.append((model_name, metrics))

if not models_to_plot:
    print("No suitable models found for horizon 1 to plot.")
else:
    # Sort models by RMSE for plotting (optional, but can be nice)
    # models_to_plot.sort(key=lambda item: item[1].get('RMSE', float('inf')))
    # For now, keeping original order from dictionary for simplicity.

    models_to_plot_count = len(models_to_plot)
    num_cols = 3
    num_rows = math.ceil(models_to_plot_count / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows), squeeze=False)
    axes = axes.flatten()
    plot_index = 0

    for model_name, metrics in models_to_plot:
        ax = axes[plot_index]
        model = metrics['model_object']
        mape = metrics.get('MAPE', float('nan'))

        X_train_for_predict = None
        X_test_for_predict = None

        # All models (including FixedEffectsLR which is now a standard LinearRegression)
        # use the X_train_scaled_df and X_test_scaled_df from the main loop for Horizon 1.
        X_train_for_predict = X_train_scaled_df
        X_test_for_predict = X_test_scaled_df
        
        predictions_train = pd.Series(dtype='float64') # Default to empty
        predictions_test = pd.Series(dtype='float64')  # Default to empty

        if X_train_for_predict is not None and not X_train_for_predict.empty:
            predictions_train = model.predict(X_train_for_predict)
        
        if X_test_for_predict is not None and not X_test_for_predict.empty:
            predictions_test = model.predict(X_test_for_predict)

        # Plotting (assumes y_train, y_test are correctly set for Horizon 1)
        if not y_train.empty:
            x_train_dates = y_train.index.get_level_values('date')
            ax.plot(x_train_dates, y_train.values, label='Actual Train', color='blue', marker='.', linestyle='', markersize=3, alpha=0.6)
            if len(predictions_train) == len(x_train_dates):
                ax.plot(x_train_dates, predictions_train, label='Predicted Train', color='orange', marker='.', linestyle='', markersize=3, alpha=0.6)

        if not y_test.empty:
            x_test_dates = y_test.index.get_level_values('date')
            ax.plot(x_test_dates, y_test.values, label='Actual Test', color='green', marker='.', linestyle='', markersize=3, alpha=0.6)
            if len(predictions_test) == len(x_test_dates):
                ax.plot(x_test_dates, predictions_test, label='Predicted Test', color='red', marker='.', linestyle='', markersize=3, alpha=0.6)

        ax.set_title(f'{model_name} (H=1) (MAPE: {mape:.2f}%)', fontsize=10)
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel(TARGET_VARIABLE, fontsize=8)
        
        if not y_train.empty or not y_test.empty:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.7)

        plot_index += 1

    # Remove empty subplots
    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=2.0)
    fig.suptitle("Model Predictions vs Actuals (Horizon 1)", fontsize=16, y=0.995) # Adjusted y for suptitle
    plt.show()
#endregion

#region Feature importance plotting
# Function to plot feature importance for a given model
def plot_feature_importance(model, feature_names, ax=None, top_n=10):
    """Plots feature importance for tree-based models or coefficients for linear models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if hasattr(model, 'feature_importances_'):
        # For tree-based models like XGBoost, DecisionTreeRegressor
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1] # Sort by importance
        
        # Select top N features
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_feature_names = np.array(feature_names)[top_indices]
        
        ax.bar(range(len(top_importances)), top_importances, align='center')
        ax.set_xticks(range(len(top_importances)))
        ax.set_xticklabels(top_feature_names, rotation=45, ha='right')
        ax.set_title('Feature Importances')
    elif hasattr(model, 'coef_'):
        # For linear models like LinearRegression, Lasso, Ridge
        coef = model.coef_
        indices = np.argsort(np.abs(coef))[::-1]  # Sort by absolute value of coefficients
        top_indices = indices[:top_n]
        top_coef = coef[top_indices]
        top_feature_names = np.array(feature_names)[top_indices]

        ax.bar(range(len(top_coef)), top_coef, align='center')
        ax.set_xticks(range(len(top_coef)))
        ax.set_xticklabels(top_feature_names, rotation=45, ha='right')
        ax.set_title('Feature Coefficients')

    ax.set_ylabel('Importance / Coefficient Value')
    plt.tight_layout()
    plt.show()

# Ensure X_train_scaled_df is available from the horizon 1 processing
# This relies on FORECAST_HORIZONS = [1] and the main loop having run.
if 1 in results_store and \
   'XGBoost' in results_store[1] and \
   results_store[1].get('XGBoost') and \
   results_store[1]['XGBoost'].get('model_object') is not None and \
   'X_train_scaled_df' in locals() and \
    X_train_scaled_df is not None and not X_train_scaled_df.empty:

    model_to_plot_fi = results_store[1]['XGBoost']['model_object']
    # Use the column names from the DataFrame the model was actually trained on
    actual_feature_names = X_train_scaled_df.columns.tolist()

    plot_feature_importance(
        model_to_plot_fi,
        actual_feature_names,
        ax=None,  # Create a new figure and axis
        top_n=10  # Specify to show only top 10 features
    )
else:
    print("Could not plot feature importance for XGBoost (H=1): Model, its training data, or feature names not available.")
#endregion