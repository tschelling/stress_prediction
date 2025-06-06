import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # For date formatting on plots
import os
import importlib

# Assuming regression_data_preparer.py is in the same directory or accessible
import regression_data_preparer
importlib.reload(regression_data_preparer) # For development, ensure latest version
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from scipy.stats import randint, uniform
from regression_data_preparer import RegressionDataPreparer

# --- Configuration ---
# This target is used by RegressionDataPreparer to generate AR lags common to all models
PRIMARY_TARGET_FOR_PREPARER = 'interest_income_to_assets' # As in 4_regressions.py

FEATURE_VARIABLES = ['gdp_qoq', 'deposit_ratio', 'loan_to_asset_ratio', 'log_total_assets', 'cpi_qoq', 'unemployment',
                     'household_delinq', 'tbill_3m', 'tbill_10y', 'spread_10y_3m', 'sp500_qoq',
                     'corp_bond_spread', 'vix_qoq']
FORECAST_HORIZONS = list(range(1, 2)) # Example: 1-quarter ahead. Modify as needed e.g. [1, 2]

# Target definitions for multi-output model
MULTI_OUTPUT_COMPONENTS = {
    'interest_income_to_assets': {'sign': +1},
    'non_interest_income_to_assets': {'sign': +1},
    'interest_expense_to_assets': {'sign': -1},
    'non_interest_expense_to_assets': {'sign': -1} # Assuming this is an expense
}
# Ensure your data.parquet contains all these columns.
# If 'non_interest_expense_to_assets' is not an expense but income, change its sign.

# Target for single-output model and overall comparison
COMPARISON_TARGET_NI = 'net_income_to_assets' # This should be ROA or similar

# DataPreparer Config (based on 4_regressions.py)
config_dict_preparer = {
    'TARGET_VARIABLE': PRIMARY_TARGET_FOR_PREPARER,
    'FEATURE_VARIABLES': FEATURE_VARIABLES,
    'INCLUDE_TIME_FE': False,
    'INCLUDE_BANK_FE': True,
    'OUTLIER_THRESHOLD_TARGET': 3.0,
    'MIN_OBS_PER_BANK': 10,
    'DATA_BEGIN': None,
    'DATA_END': None,
    'RESTRICT_TO_NUMBER_OF_BANKS': 2000,
    'RESTRICT_TO_BANK_SIZE': None,
    'RESTRICT_TO_MINIMAL_DEPOSIT_RATIO': None,
    'RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO': 0.1,
    'INCLUDE_AUTOREGRESSIVE_LAGS': True,
    'NUMBER_OF_LAGS_TO_INCLUDE': 4,
    'TRAIN_TEST_SPLIT_DIMENSION': 'date',
    'TEST_SPLIT': 0.2
}

# XGBoost default parameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist' # Faster for larger datasets
}

# Hyperparameter Tuning Configuration (similar to 4_regressions.py)
N_SPLITS_CV = 3  # Number of splits for TimeSeriesSplit
USE_RANDOM_SEARCH_CV = True # True for RandomizedSearchCV, False for GridSearchCV
N_ITER_RANDOM_SEARCH = 30   # Number of parameter settings that are sampled for RandomizedSearchCV (100 in 4_regressions.py, using 20 for speed here)

# XGBoost Parameter Grid for RandomizedSearchCV
XGB_PARAM_GRID_RANDOM = {
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.01, 0.29), # Max is 0.01 + 0.29 = 0.3
    'max_depth': randint(3, 10),
    'subsample': uniform(0.6, 0.4), # Samples from 0.6 to 1.0
    'colsample_bytree': uniform(0.6, 0.4) # Samples from 0.6 to 1.0
}

# --- Helper function to prepare y for a specific target and horizon ---
def prepare_y_for_target(base_data_df: pd.DataFrame, target_col_name: str, horizon: int,
                         X_train_idx: pd.MultiIndex, X_test_idx: pd.MultiIndex = None):
    """
    Prepares the target variable (y) for a specific column name and horizon.
    Shifts the target, drops NaNs, and aligns with provided X_train and X_test indices.
    """
    df_h = base_data_df.copy()
    shifted_target_col = f'{target_col_name}_target_h{horizon}'

    if target_col_name not in df_h.columns:
        print(f"Warning: Target column '{target_col_name}' not found in base data for y preparation.")
        empty_idx_name = X_train_idx.names if X_train_idx is not None else ['id', 'date']
        empty_series = pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=empty_idx_name))
        return empty_series, empty_series

    df_h[shifted_target_col] = df_h.groupby(level='id', group_keys=False)[target_col_name].shift(-horizon)
    df_h.dropna(subset=[shifted_target_col], inplace=True)

    if df_h.empty:
        print(f"Warning: DataFrame empty after shifting and NaN drop for target '{target_col_name}', H{horizon}.")
        empty_idx_name = X_train_idx.names if X_train_idx is not None else ['id', 'date']
        empty_series = pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=empty_idx_name))
        return empty_series, empty_series

    y_full_specific = df_h[shifted_target_col]

    y_train_specific = pd.Series(dtype='float64')
    y_test_specific = pd.Series(dtype='float64')

    if X_train_idx is not None and not X_train_idx.empty:
        y_train_specific = y_full_specific.reindex(X_train_idx).dropna()
    else:
        y_train_specific = pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=y_full_specific.index.names))

    if X_test_idx is not None and not X_test_idx.empty:
        y_test_specific = y_full_specific.reindex(X_test_idx).dropna()
    else:
         y_test_specific = pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=y_full_specific.index.names))

    return y_train_specific, y_test_specific

# --- Plotting function ---
def plot_aggregated_timeseries_comparison(fig, ax,
                                          y_train_true_series, y_train_pred_series,
                                          y_test_true_series, y_test_pred_series,
                                          model_name_title, target_name_label):
    """
    Plots aggregated (median and IQR) time series for actual vs. predicted values
    for both training and test sets on the same axes.
    """

    plot_sets = [
        (y_train_true_series, 'Actual Train', 'blue', '-'),
        (y_train_pred_series, 'Predicted Train', 'deepskyblue', '--'),
        (y_test_true_series, 'Actual Test', 'green', '-'),
        (y_test_pred_series, 'Predicted Test', 'red', '--')
    ]

    any_data_plotted = False
    for series_data, label_prefix, color, style in plot_sets:
        if series_data is None or series_data.empty:
            print(f"Data for '{label_prefix}' for '{target_name_label}' in '{model_name_title}' is empty. Skipping.")
            continue

        if not isinstance(series_data.index, pd.MultiIndex) or 'date' not in series_data.index.names:
            print(f"Skipping {label_prefix} for {model_name_title}: index is not MultiIndex with 'date'.")
            continue
        try:
            # Ensure data is numeric before quantile calculation
            numeric_series = pd.to_numeric(series_data, errors='coerce').dropna()
            if numeric_series.empty:
                print(f"Warning: Series for {label_prefix} is empty after converting to numeric. Skipping plot.")
                continue

            agg_data = numeric_series.groupby(level='date').agg(
                q25=lambda x: x.quantile(0.25),
                q50=lambda x: x.quantile(0.50),
                q75=lambda x: x.quantile(0.75)
            )
            if agg_data.empty:
                print(f"Warning: Empty aggregated data for {label_prefix}. Skipping plot.")
                continue
            ax.plot(agg_data.index, agg_data['q50'], label=f'{label_prefix} Median', color=color, linestyle=style)
            ax.fill_between(agg_data.index, agg_data['q25'], agg_data['q75'], color=color, alpha=0.2) # No label for fill to avoid duplicate legend items
            any_data_plotted = True
        except Exception as e:
            print(f"Error during aggregation/plotting for {label_prefix} ({target_name_label}): {e}")

    if not any_data_plotted:
        ax.text(0.5, 0.5, "No data available to plot", ha='center', va='center', transform=ax.transAxes)
        # Set title even if no data, so the subplot is identifiable

    ax.set_title(f"{model_name_title}\n{target_name_label} (Train & Test)")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_name_label)
    ax.legend(fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.7)
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        if fig: fig.autofmt_xdate(rotation=45)
    except Exception as e_date:
        print(f"Warning: Could not format x-axis as dates: {e_date}")

# --- Main Script ---
def main():
    print("--- Loading Data ---")
    try:
        raw_data = pd.read_parquet('data.parquet')
        print(f"Raw data loaded. Shape: {raw_data.shape}")
    except Exception as e:
        print(f"Error loading data.parquet: {e}. Please ensure the file exists and is valid.")
        return

    # Check if all required component columns exist in raw_data
    missing_component_cols = [col for col in MULTI_OUTPUT_COMPONENTS.keys() if col not in raw_data.columns]
    if missing_component_cols:
        print(f"Error: The following component columns are missing in data.parquet: {missing_component_cols}. Exiting.")
        return
    if COMPARISON_TARGET_NI not in raw_data.columns:
        print(f"Error: The comparison target column '{COMPARISON_TARGET_NI}' is missing in data.parquet. Exiting.")
        return

    data_preparer = RegressionDataPreparer(initial_df=raw_data, config=config_dict_preparer)
    if data_preparer.base_data_for_horizons is None or data_preparer.base_data_for_horizons.empty:
        print("Base data preparation by RegressionDataPreparer failed or resulted in empty data. Exiting.")
        return
    base_data_for_all_targets = data_preparer.base_data_for_horizons.copy()

    comparison_metrics = {}
    plotting_data_store = {} # To store {horizon: {plot_key: {train_actual, train_pred, test_actual, test_pred}}}

    for horizon in FORECAST_HORIZONS:
        print(f"\n--- Processing Horizon: {horizon} ---")
        comparison_metrics[horizon] = {}
        plotting_data_store[horizon] = {}

        # Get X data (scaled) using the primary target from config
        # The y_train, y_test from get_horizon_specific_data are for PRIMARY_TARGET_FOR_PREPARER,
        # we will prepare our specific y's manually.
        prepared_data_primary = data_preparer.get_horizon_specific_data(horizon=horizon)
        if prepared_data_primary is None:
            print(f"  Failed to prepare X data for horizon {horizon}. Skipping this horizon.")
            continue
        X_train_scaled_df, X_test_scaled_df, _, _, _, _ = prepared_data_primary

        if X_train_scaled_df is None or X_train_scaled_df.empty:
            print(f"  X_train_scaled_df is empty for horizon {horizon}. Skipping this horizon.")
            continue
        
        if X_test_scaled_df is None: # Ensure X_test_scaled_df is not None for safety
            X_test_scaled_df = pd.DataFrame(columns=X_train_scaled_df.columns,
                                            index=pd.MultiIndex.from_tuples([], names=X_train_scaled_df.index.names))

        # --- 1. Multi-output XGBoost for Components ---
        print("  Training Multi-output XGBoost for components...")
        y_train_multi_list = []
        valid_multi_targets_for_training = []

        for target_name in MULTI_OUTPUT_COMPONENTS.keys():
            y_train_component, _ = prepare_y_for_target(
                base_data_for_all_targets, target_name, horizon, X_train_scaled_df.index
            )
            if not y_train_component.empty:
                y_train_multi_list.append(y_train_component)
                valid_multi_targets_for_training.append(target_name)
            else:
                print(f"    Skipping component '{target_name}' for multi-output training due to empty y_train.")

        if len(y_train_multi_list) > 0:
            y_train_multi_df_unaligned = pd.concat(y_train_multi_list, axis=1)
            y_train_multi_df_unaligned.columns = valid_multi_targets_for_training
            
            common_idx_train_multi = X_train_scaled_df.index.intersection(y_train_multi_df_unaligned.index)
            X_train_scaled_multi = X_train_scaled_df.loc[common_idx_train_multi]
            y_train_multi_aligned = y_train_multi_df_unaligned.loc[common_idx_train_multi]

            if not X_train_scaled_multi.empty and not y_train_multi_aligned.empty:
                base_xgb_multi = xgb.XGBRegressor(**XGB_PARAMS)
                
                if len(X_train_scaled_multi) >= N_SPLITS_CV + 1:
                    print(f"    Tuning Multi-output XGBoost with {'RandomizedSearchCV' if USE_RANDOM_SEARCH_CV else 'GridSearchCV'}...")
                    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
                    
                    if USE_RANDOM_SEARCH_CV:
                        search_cv_multi = RandomizedSearchCV(
                            estimator=base_xgb_multi,
                            param_distributions=XGB_PARAM_GRID_RANDOM,
                            n_iter=N_ITER_RANDOM_SEARCH,
                            cv=tscv,
                            scoring='neg_mean_squared_error', # Averaged for multi-output
                            random_state=42,
                            n_jobs=-1,
                            verbose=1 # Set to 0 for less output
                        )
                    else: # GridSearchCV - requires a grid of specific values, not distributions
                        # For GridSearchCV, you'd define XGB_PARAM_GRID_GRID with lists of values
                        print("    GridSearchCV not fully configured with discrete grid, using default XGB_PARAMS for base model.")
                        search_cv_multi = base_xgb_multi # Fallback to base if grid not set for GridSearchCV

                    search_cv_multi.fit(X_train_scaled_multi, y_train_multi_aligned)
                    model_multi_xgb = search_cv_multi.best_estimator_
                    print(f"    Best params for Multi-Output XGBoost: {search_cv_multi.best_params_ if hasattr(search_cv_multi, 'best_params_') else 'N/A (using base)'}")
                else:
                    print(f"    Not enough samples ({len(X_train_scaled_multi)}) for {N_SPLITS_CV} CV splits. Fitting Multi-output XGBoost with default parameters.")
                    model_multi_xgb = base_xgb_multi
                    model_multi_xgb.fit(X_train_scaled_multi, y_train_multi_aligned)
                print(f"    Multi-output XGBoost fitting complete for {len(valid_multi_targets_for_training)} components: {valid_multi_targets_for_training}")

                # Predictions (Train) for components
                preds_multi_train_all_comps = model_multi_xgb.predict(X_train_scaled_multi)
                preds_multi_train_df_comps = pd.DataFrame(preds_multi_train_all_comps, columns=valid_multi_targets_for_training, index=X_train_scaled_multi.index)

                # Predictions (Test) for components
                y_test_multi_list = []
                for target_name in valid_multi_targets_for_training:
                     _, y_test_component = prepare_y_for_target(
                         base_data_for_all_targets, target_name, horizon, X_train_scaled_df.index, X_test_scaled_df.index
                     )
                     y_test_multi_list.append(y_test_component)

                preds_multi_test_df_comps = pd.DataFrame(columns=valid_multi_targets_for_training) # Default empty
                y_test_multi_aligned = pd.DataFrame(columns=valid_multi_targets_for_training) # Default empty

                if y_test_multi_list:
                    y_test_multi_df_unaligned = pd.concat(y_test_multi_list, axis=1)
                    y_test_multi_df_unaligned.columns = valid_multi_targets_for_training

                    common_idx_test_multi = X_test_scaled_df.index.intersection(y_test_multi_df_unaligned.index)
                    X_test_scaled_multi_eval = X_test_scaled_df.loc[common_idx_test_multi]
                    y_test_multi_aligned = y_test_multi_df_unaligned.loc[common_idx_test_multi].dropna(how='all')

                    if not X_test_scaled_multi_eval.empty and not y_test_multi_aligned.empty:
                        preds_multi_test_all_comps = model_multi_xgb.predict(X_test_scaled_multi_eval)
                        preds_multi_test_df_comps = pd.DataFrame(preds_multi_test_all_comps, columns=valid_multi_targets_for_training, index=X_test_scaled_multi_eval.index)

                # Store individual component plots
                for comp_name in valid_multi_targets_for_training:
                    plot_key_comp = f"Comp_{comp_name}_Multi"
                    plotting_data_store[horizon][plot_key_comp] = {
                        'y_train_actual': y_train_multi_aligned[comp_name].copy() if comp_name in y_train_multi_aligned else pd.Series(dtype='float64'),
                        'y_train_pred': preds_multi_train_df_comps[comp_name].copy() if comp_name in preds_multi_train_df_comps else pd.Series(dtype='float64'),
                        'y_test_actual': y_test_multi_aligned[comp_name].copy() if comp_name in y_test_multi_aligned and not y_test_multi_aligned.empty else pd.Series(dtype='float64'),
                        'y_test_pred': preds_multi_test_df_comps[comp_name].copy() if comp_name in preds_multi_test_df_comps and not preds_multi_test_df_comps.empty else pd.Series(dtype='float64'),
                        'target_label': comp_name
                    }
                
                # Aggregate component predictions for COMPARISON_TARGET_NI
                pred_nia_from_multi_train = pd.Series(0.0, index=preds_multi_train_df_comps.index)
                pred_nia_from_multi_test = pd.Series(0.0, index=preds_multi_test_df_comps.index)

                for comp_name, comp_info in MULTI_OUTPUT_COMPONENTS.items():
                    if comp_name in preds_multi_train_df_comps:
                        pred_nia_from_multi_train += comp_info['sign'] * preds_multi_train_df_comps[comp_name]
                    if comp_name in preds_multi_test_df_comps:
                        pred_nia_from_multi_test += comp_info['sign'] * preds_multi_test_df_comps[comp_name]
                
                # Evaluate aggregated prediction
                y_train_nia_actual, y_test_nia_actual = prepare_y_for_target(
                    base_data_for_all_targets, COMPARISON_TARGET_NI, horizon, X_train_scaled_df.index, X_test_scaled_df.index
                )
                
                if not pred_nia_from_multi_test.empty and not y_test_nia_actual.empty:
                    common_idx_agg_test = pred_nia_from_multi_test.index.intersection(y_test_nia_actual.index)
                    pred_nia_from_multi_test_aligned = pred_nia_from_multi_test.loc[common_idx_agg_test]
                    y_test_nia_actual_aligned = y_test_nia_actual.loc[common_idx_agg_test]

                    if not pred_nia_from_multi_test_aligned.empty and not y_test_nia_actual_aligned.empty:
                        rmse_nia_multi = np.sqrt(mean_squared_error(y_test_nia_actual_aligned, pred_nia_from_multi_test_aligned))
                        comparison_metrics[horizon]['NI_AggregatedMultiOutput_RMSE'] = rmse_nia_multi
                        print(f"    RMSE for {COMPARISON_TARGET_NI} (Aggregated Multi-output): {rmse_nia_multi:.4f}")
                        
                        # Store data for plotting NI from multi-output aggregation
                        plot_key_ni_multi = "NI_Aggregated_Multi"
                        plotting_data_store[horizon][plot_key_ni_multi] = {'y_test_actual': y_test_nia_actual_aligned, 
                                                                           'y_test_pred': pred_nia_from_multi_test_aligned,
                                                                           'target_label': COMPARISON_TARGET_NI}
                        if not pred_nia_from_multi_train.empty and not y_train_nia_actual.empty:
                            common_idx_agg_train = pred_nia_from_multi_train.index.intersection(y_train_nia_actual.index)
                            plotting_data_store[horizon][plot_key_ni_multi]['y_train_actual'] = y_train_nia_actual.loc[common_idx_agg_train]
                            plotting_data_store[horizon][plot_key_ni_multi]['y_train_pred'] = pred_nia_from_multi_train.loc[common_idx_agg_train]
            else:
                print("    Training data for multi-output XGBoost is empty after alignment. Skipping.")
        else:
            print("    Not enough valid components to train multi-output XGBoost. Skipping.")

        # --- 2. Single-output XGBoost for COMPARISON_TARGET_NI ---
        print(f"  Training Single-output XGBoost for {COMPARISON_TARGET_NI}...")
        y_train_ni_direct, y_test_ni_direct = prepare_y_for_target(
            base_data_for_all_targets, COMPARISON_TARGET_NI, horizon, X_train_scaled_df.index, X_test_scaled_df.index
        )
        if not y_train_ni_direct.empty:
            common_idx_train_ni_direct = X_train_scaled_df.index.intersection(y_train_ni_direct.index)
            X_train_scaled_single_ni = X_train_scaled_df.loc[common_idx_train_ni_direct]
            y_train_ni_direct_aligned = y_train_ni_direct.loc[common_idx_train_ni_direct]

            if not X_train_scaled_single_ni.empty and not y_train_ni_direct_aligned.empty:
                base_xgb_single_ni = xgb.XGBRegressor(**XGB_PARAMS)

                if len(X_train_scaled_single_ni) >= N_SPLITS_CV + 1:
                    print(f"    Tuning Single-output XGBoost for {COMPARISON_TARGET_NI} with {'RandomizedSearchCV' if USE_RANDOM_SEARCH_CV else 'GridSearchCV'}...")
                    tscv_single = TimeSeriesSplit(n_splits=N_SPLITS_CV)

                    if USE_RANDOM_SEARCH_CV:
                        search_cv_single_ni = RandomizedSearchCV(
                            estimator=base_xgb_single_ni,
                            param_distributions=XGB_PARAM_GRID_RANDOM,
                            n_iter=N_ITER_RANDOM_SEARCH,
                            cv=tscv_single,
                            scoring='neg_mean_squared_error',
                            random_state=42,
                            n_jobs=-1,
                            verbose=1
                        )
                    else: # GridSearchCV
                        print("    GridSearchCV not fully configured with discrete grid, using default XGB_PARAMS for base model.")
                        search_cv_single_ni = base_xgb_single_ni # Fallback

                    search_cv_single_ni.fit(X_train_scaled_single_ni, y_train_ni_direct_aligned)
                    model_single_ni = search_cv_single_ni.best_estimator_
                    print(f"    Best params for Single-Output XGBoost ({COMPARISON_TARGET_NI}): {search_cv_single_ni.best_params_ if hasattr(search_cv_single_ni, 'best_params_') else 'N/A (using base)'}")
                else:
                    print(f"    Not enough samples ({len(X_train_scaled_single_ni)}) for {N_SPLITS_CV} CV splits. Fitting Single-output XGBoost ({COMPARISON_TARGET_NI}) with default parameters.")
                    model_single_ni = base_xgb_single_ni
                    model_single_ni.fit(X_train_scaled_single_ni, y_train_ni_direct_aligned)

                preds_single_ni_train = pd.Series(model_single_ni.predict(X_train_scaled_single_ni), index=y_train_ni_direct_aligned.index)
                
                preds_single_ni_test = pd.Series(dtype='float64')
                y_test_ni_direct_aligned = pd.Series(dtype='float64')

                if not y_test_ni_direct.empty:
                    common_idx_test_ni_direct = X_test_scaled_df.index.intersection(y_test_ni_direct.index)
                    X_test_scaled_single_ni_eval = X_test_scaled_df.loc[common_idx_test_ni_direct]
                    y_test_ni_direct_aligned = y_test_ni_direct.loc[common_idx_test_ni_direct]

                    if not X_test_scaled_single_ni_eval.empty and not y_test_ni_direct_aligned.empty:
                        preds_single_ni_test = pd.Series(model_single_ni.predict(X_test_scaled_single_ni_eval), index=y_test_ni_direct_aligned.index)
                        rmse_ni_single = np.sqrt(mean_squared_error(y_test_ni_direct_aligned, preds_single_ni_test))
                        comparison_metrics[horizon]['NI_SingleOutput_RMSE'] = rmse_ni_single
                        print(f"    RMSE for {COMPARISON_TARGET_NI} (Direct Single-output): {rmse_ni_single:.4f}")
                
                plot_key_ni_single = "NI_Direct_Single"
                plotting_data_store[horizon][plot_key_ni_single] = {
                    'y_train_actual': y_train_ni_direct_aligned,
                    'y_train_pred': preds_single_ni_train,
                    'y_test_actual': y_test_ni_direct_aligned,
                    'y_test_pred': preds_single_ni_test,
                    'target_label': COMPARISON_TARGET_NI
                }
            else:
                print(f"    Training data for {COMPARISON_TARGET_NI} (Single-output) is empty after alignment. Skipping.")
        else:
            print(f"    y_train for {COMPARISON_TARGET_NI} (Single-output) is empty. Skipping.")

    # --- Display comparison results ---
    print(f"\n--- Comparison of RMSE for {COMPARISON_TARGET_NI} ---")
    for h, metrics in comparison_metrics.items():
        print(f"  Horizon {h}:")
        if 'NI_AggregatedMultiOutput_RMSE' in metrics:
            print(f"    {COMPARISON_TARGET_NI} from Aggregated Multi-Output RMSE: {metrics['NI_AggregatedMultiOutput_RMSE']:.4f}")
        if 'NI_SingleOutput_RMSE' in metrics:
            print(f"    {COMPARISON_TARGET_NI} from Direct Single-Output RMSE:   {metrics['NI_SingleOutput_RMSE']:.4f}")

    # --- Visualization ---
    print("\n--- Plotting Results ---")
    plot_items_to_visualize = []
    for h_plot, data_for_h in plotting_data_store.items():
        # Plot aggregated NI from multi-output
        if "NI_Aggregated_Multi" in data_for_h:
            plot_items_to_visualize.append({
                'data_dict': data_for_h["NI_Aggregated_Multi"],
                'model_title': f'{COMPARISON_TARGET_NI} (Aggregated Multi-Output) H{h_plot}',
                'target_label': data_for_h["NI_Aggregated_Multi"].get('target_label', COMPARISON_TARGET_NI)
            })
        # Plot direct NI from single-output
        if "NI_Direct_Single" in data_for_h:
            plot_items_to_visualize.append({
                'data_dict': data_for_h["NI_Direct_Single"],
                'model_title': f'{COMPARISON_TARGET_NI} (Direct Single-Output) H{h_plot}',
                'target_label': data_for_h["NI_Direct_Single"].get('target_label', COMPARISON_TARGET_NI)
            })
        # Plot individual components
        for comp_name in MULTI_OUTPUT_COMPONENTS.keys():
            plot_key_comp = f"Comp_{comp_name}_Multi"
            if plot_key_comp in data_for_h:
                 plot_items_to_visualize.append({
                    'data_dict': data_for_h[plot_key_comp],
                    'model_title': f'{comp_name} (from Multi-Output) H{h_plot}',
                    'target_label': data_for_h[plot_key_comp].get('target_label', comp_name)
                })
    
    if not plot_items_to_visualize:
        print("No data available for plotting.")
        return

    num_main_plots = len(plot_items_to_visualize)
    num_cols_viz = 1  # Train and Test plots combined in one column
    num_rows_viz = num_main_plots

    if num_main_plots == 0:
        print("No items to visualize.")
        return
        
    fig, axes = plt.subplots(num_rows_viz, num_cols_viz, figsize=(12, 5 * num_rows_viz), squeeze=False) # Adjusted figsize width for single column
    
    for i, item in enumerate(plot_items_to_visualize):
        data_dict = item['data_dict']
        model_title = item['model_title']
        target_label = item['target_label']

        ax_combined = axes[i, 0] # Only one column now
        plot_aggregated_timeseries_comparison(fig, ax_combined,
                                              data_dict.get('y_train_actual'), data_dict.get('y_train_pred'),
                                              data_dict.get('y_test_actual'), data_dict.get('y_test_pred'),
                                              model_title, target_label)

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust for suptitle
    fig.suptitle("Model Predictions vs. Actuals (Median and IQR over Banks)", fontsize=16, y=0.99)
    plt.show()

if __name__ == "__main__":
    # Create a dummy data.parquet for testing if it doesn't exist
    if not os.path.exists('data.parquet'):
        print("Creating dummy data.parquet for testing purposes...")
        rng = np.random.default_rng(42)
        n_banks = 20
        n_dates_per_bank = 40
        dates = pd.to_datetime(pd.date_range('2010-01-01', periods=n_dates_per_bank, freq='Q'))
        ids = [f'bank_{i}' for i in range(n_banks)]
        multi_index = pd.MultiIndex.from_product([ids, dates], names=['id', 'date'])
        
        dummy_data = pd.DataFrame(index=multi_index)
        
        # Financial Ratios (components of NI)
        dummy_data['interest_income_to_assets'] = rng.normal(0.03, 0.01, size=len(multi_index))
        dummy_data['non_interest_income_to_assets'] = rng.normal(0.01, 0.005, size=len(multi_index))
        dummy_data['interest_expense_to_assets'] = rng.normal(0.01, 0.005, size=len(multi_index))
        dummy_data['non_interest_expense_to_assets'] = rng.normal(0.005, 0.002, size=len(multi_index))
        
        # Net Income (calculated from components)
        dummy_data[COMPARISON_TARGET_NI] = (
            dummy_data['interest_income_to_assets'] +
            dummy_data['non_interest_income_to_assets'] -
            dummy_data['interest_expense_to_assets'] -
            dummy_data['non_interest_expense_to_assets'] +
            rng.normal(0, 0.001, size=len(multi_index)) # Add some noise
        )
        
        # Feature Variables
        for feature in FEATURE_VARIABLES:
            if feature == 'log_total_assets':
                 dummy_data[feature] = rng.normal(15, 2, size=len(multi_index))
            elif feature == 'deposit_ratio' or feature == 'loan_to_asset_ratio':
                 dummy_data[feature] = rng.uniform(0.4, 0.9, size=len(multi_index))
            else: # Macro vars
                 # For macro vars, make them vary by date but same for all banks at a given date
                 macro_values = rng.normal(0, 0.1, size=len(dates))
                 dummy_data[feature] = dummy_data.index.get_level_values('date').map(pd.Series(macro_values, index=dates))

        dummy_data.to_parquet('data.parquet')
        print("Dummy data.parquet created.")

    main()
    print("--- Script Finished ---")
