import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import math
from sklearn.pipeline import Pipeline
from IPython.display import display
from publication_name_mapping import PUBLICATION_NAMES

# --- Configuration ---
ARTIFACTS_BASE_DIR = "models_and_results1"  # Should match 2_regressions.py
TOP_N_FEATURES = 10  # For feature importance plots
MODELS_FOR_FI_PLOT = ["XGBoost", "RandomForest", "DecisionTree"]

# Suppress TensorFlow INFO and WARNING messages if Keras models are loaded

# --- Helper Functions (copied from 2_regressions.py) ---

def plot_aggregated_timeseries(ax, data_series, label_prefix, color, line_style='-'):
    if not isinstance(data_series.index, pd.MultiIndex) or 'date' not in data_series.index.names:
        print(f"Skipping {label_prefix} plot: data index is not a MultiIndex with 'date'. Index type: {type(data_series.index)}")
        return
    if data_series.empty:
        print(f"Skipping {label_prefix} plot: data series is empty.")
        return

    # Ensure 'date' level is datetime for proper groupby and plotting
    temp_series = data_series.copy()
    if 'date' in temp_series.index.names:
        temp_series.index = temp_series.index.set_levels(pd.to_datetime(temp_series.index.levels[temp_series.index.names.index('date')]), level='date')

    quantiles_series = temp_series.groupby(level='date').quantile([0.25, 0.50, 0.75])
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

    ax.plot(dates_idx, mean_values, label=f'{label_prefix} Median', color=color, linestyle=line_style) # Changed Mean to Median as per quantile
    ax.fill_between(dates_idx, q1_values, q3_values, color=color, alpha=0.2, label=f'{label_prefix} IQR')

def plot_feature_importance(model, feature_names, ax=None, top_n=10, r2_score_val=None, model_display_name=None):
    if ax is None:
        fig_fi, ax = plt.subplots(figsize=(10, 6))

    title_prefix = model_display_name if model_display_name else "Feature"
    title_suffix = ""
    if r2_score_val is not None and not np.isnan(r2_score_val):
        title_suffix = f" (R²: {r2_score_val:.2f})"

    importances_values = None
    processed_feature_names = feature_names

    if isinstance(model, Pipeline) and 'rfe' in model.named_steps:
        rfe_step = model.named_steps['rfe']
        # Get the name of the final estimator in the pipeline
        final_estimator_name = [name for name in model.named_steps if name != 'rfe'][0]
        regressor_step = model.named_steps[final_estimator_name]

        if hasattr(rfe_step, 'support_'):
            processed_feature_names = np.array(feature_names)[rfe_step.support_]
            # print(f"  Features selected by RFE for {model.named_steps[final_estimator_name].__class__.__name__}: {list(processed_feature_names)}")
            if hasattr(regressor_step, 'feature_importances_'):
                importances_values = regressor_step.feature_importances_
            elif hasattr(regressor_step, 'coef_'):
                importances_values = regressor_step.coef_
            ax.set_title(f'{title_prefix} Importances (RFE w/ {type(regressor_step).__name__}){title_suffix}', fontsize=9)
        else:
            # print(f"  RFE step in pipeline for {model.named_steps[final_estimator_name].__class__.__name__} has not been fitted or does not have 'support_' attribute.")
            ax.text(0.5, 0.5, "RFE step not fitted", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{title_prefix} (RFE - not fit){title_suffix}', fontsize=9)
            return
    elif hasattr(model, 'feature_importances_'):
        importances_values = model.feature_importances_
        ax.set_title(f'{title_prefix} Importances{title_suffix}', fontsize=9)
    elif hasattr(model, 'coef_'):
        importances_values = model.coef_
        ax.set_title(f'{title_prefix} Coefficients{title_suffix}', fontsize=9)

    if importances_values is not None:
        if len(importances_values.shape) > 1 and importances_values.shape[0] == 1:
            importances_values = importances_values.flatten()

        if len(importances_values) != len(processed_feature_names):
            # print(f"Warning: Mismatch in length of importances ({len(importances_values)}) and feature names ({len(processed_feature_names)}) for {model_display_name}. Skipping feature importance plot.")
            ax.text(0.5, 0.5, "Importance/feature name mismatch", ha='center', va='center', transform=ax.transAxes)
            return

        indices = np.argsort(np.abs(importances_values))[::-1]
        top_indices = indices[:top_n]
        top_importances = importances_values[top_indices]
        top_feature_names_raw = np.array(processed_feature_names)[top_indices]
        top_feature_names = [PUBLICATION_NAMES.get(name, name) for name in top_feature_names_raw]
        
        ax.bar(range(len(top_importances)), top_importances, align='center')
        ax.set_xticks(range(len(top_importances)))
        ax.set_xticklabels(top_feature_names, rotation=60, ha='right', fontsize=7)
        ax.set_ylabel('Importance / Coefficient Value', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
    else:
        ax.text(0.5, 0.5, "Feature importance not available", ha='center', va='center', transform=ax.transAxes)
        if not (isinstance(model, Pipeline) and 'rfe' in model.named_steps):
            ax.set_title(f'{title_prefix} Importance N/A{title_suffix}', fontsize=9)

# --- Metric Calculation Helper (copied from 2_regressions.py and adapted) ---
def _calculate_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict:
    """Calculates standard regression metrics."""
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Local import

    if y_true.empty or predictions is None or len(predictions) != len(y_true):
        return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan}

    # Ensure predictions is a 1D array if it's not already (e.g. Keras might return (N,1))
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    
    # Align predictions with y_true if y_true is a Series with an index
    # and predictions is a numpy array that needs to be converted to a Series
    if isinstance(y_true, pd.Series) and isinstance(predictions, np.ndarray):
        if len(predictions) == len(y_true.index):
            predictions_series = pd.Series(predictions, index=y_true.index)
        else: # Length mismatch, cannot safely align
            print(f"Warning: Length mismatch between y_true ({len(y_true)}) and predictions ({len(predictions)}). Cannot align for metrics.")
            return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan}
    elif isinstance(predictions, pd.Series):
        predictions_series = predictions
    else: # Both are numpy arrays or other unhandled types
        predictions_series = pd.Series(predictions) # Convert to series for consistency if y_true is also array

    mae = mean_absolute_error(y_true, predictions_series)
    mse = mean_squared_error(y_true, predictions_series)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions_series)

    y_true_no_zeros = y_true.replace(0, np.nan).dropna()
    mape = np.nan
    if not y_true_no_zeros.empty:
        predictions_aligned_for_mape = predictions_series.loc[y_true_no_zeros.index]
        mape = np.mean(np.abs((y_true_no_zeros - predictions_aligned_for_mape) / y_true_no_zeros)) * 100
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape} 

# New helper functions for data characteristics
def load_results_store(artifacts_dir: str) -> dict:
    """Loads the results_store dictionary."""
    path = os.path.join(artifacts_dir, "results_store.joblib")
    return joblib.load(path)

def load_data_for_horizon(artifacts_dir: str, target_variable_name: str, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads the scaled training and testing feature sets (X) and target variables (y)
    for a specific target variable and forecast horizon.
    """
    target_specific_artifact_dir = os.path.join(artifacts_dir, f"target_{target_variable_name}")
    data_dir = os.path.join(target_specific_artifact_dir, f"horizon_{horizon}", "data")

    X_train_path = os.path.join(data_dir, "X_train_scaled.parquet")
    X_test_path = os.path.join(data_dir, "X_test_scaled.parquet")
    y_train_path = os.path.join(data_dir, "y_train.parquet")
    y_test_path = os.path.join(data_dir, "y_test.parquet")

    X_train_scaled_df = pd.read_parquet(X_train_path)
    X_test_scaled_df = pd.read_parquet(X_test_path)
    y_train_df = pd.read_parquet(y_train_path)
    y_test_df = pd.read_parquet(y_test_path)
    
    y_train = y_train_df[y_train_df.columns[0]]
    y_test = y_test_df[y_test_df.columns[0]]

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test

def get_data_characteristics_per_model(artifacts_base_dir: str) -> pd.DataFrame:
    """
    Generates a DataFrame listing the number of unique banks, dates, and total
    observations for the training and testing sets, broken down by target variable,
    forecast horizon, and model.
    """
    results_store = load_results_store(artifacts_base_dir)
    all_data_characteristics = []

    for target_variable, horizons_data in results_store.items():
        for horizon_val, models_data in horizons_data.items():
            X_train_scaled_df, X_test_scaled_df, y_train, y_test = load_data_for_horizon(artifacts_base_dir, target_variable, horizon_val)
            train_unique_banks = X_train_scaled_df.index.get_level_values('id').nunique()
            train_unique_dates = X_train_scaled_df.index.get_level_values('date').nunique()
            train_observations = len(X_train_scaled_df)
            
            test_unique_banks = X_test_scaled_df.index.get_level_values('id').nunique()
            
            total_unique_banks = pd.Series(X_train_scaled_df.index.get_level_values('id').tolist() + X_test_scaled_df.index.get_level_values('id').tolist()).nunique()
            test_unique_dates = X_test_scaled_df.index.get_level_values('date').nunique()
            test_observations = len(X_test_scaled_df)
            for model_name in models_data.keys():
                all_data_characteristics.append({'Target Variable': target_variable, 'Horizon': horizon_val, 'Model Name': model_name, 'Train_Unique_Banks': train_unique_banks, 'Train_Unique_Dates': train_unique_dates, 'Train_Observations': train_observations, 'Test_Unique_Banks': test_unique_banks, 'Test_Unique_Dates': test_unique_dates, 'Test_Observations': test_observations, 'Total_Unique_Banks': total_unique_banks})
    return pd.DataFrame(all_data_characteristics)



if not os.path.exists(ARTIFACTS_BASE_DIR):
    print(f"Artifacts directory not found: {ARTIFACTS_BASE_DIR}")
    exit()

all_calculated_metrics = [] # To store metrics for all models, targets, horizons

for target_dir_name in os.listdir(ARTIFACTS_BASE_DIR):
    if target_dir_name.startswith("target_"):
        current_target_variable = target_dir_name.replace("target_", "")
        pub_target_name = PUBLICATION_NAMES.get(current_target_variable, current_target_variable)
        target_path = os.path.join(ARTIFACTS_BASE_DIR, target_dir_name)

        print(f"\n--- Inspecting Target Variable: {pub_target_name} ---")

        for horizon_dir_name in os.listdir(target_path):
            if horizon_dir_name.startswith("horizon_"):
                horizon_val_str = horizon_dir_name.replace("horizon_", "")
                horizon_val = int(horizon_val_str)
                current_artifact_dir = os.path.join(target_path, horizon_dir_name)
                data_dir = os.path.join(current_artifact_dir, "data")

                print(f"  -- Horizon: {horizon_val} --")

                # Load data
                X_train_path = os.path.join(data_dir, "X_train_scaled.parquet")
                X_test_path = os.path.join(data_dir, "X_test_scaled.parquet")
                y_train_path = os.path.join(data_dir, "y_train.parquet")
                y_test_path = os.path.join(data_dir, "y_test.parquet")

                if not all(map(os.path.exists, [X_train_path, X_test_path, y_train_path, y_test_path])):
                    print(f"    Skipping horizon {horizon_val} for target {current_target_variable}: Missing one or more data files in {data_dir}")
                    continue

                X_train_scaled_df = pd.read_parquet(X_train_path)
                X_test_scaled_df = pd.read_parquet(X_test_path)
                y_train_df = pd.read_parquet(y_train_path)
                y_test_df = pd.read_parquet(y_test_path)
                
                y_train = y_train_df[y_train_df.columns[0]]
                y_test = y_test_df[y_test_df.columns[0]]
                
                feature_names = X_train_scaled_df.columns.tolist()

                # --- Plot Predictions vs Actuals ---
                models_to_plot_pred = []
                for item_name in os.listdir(current_artifact_dir):
                    if item_name.endswith(".joblib") and item_name != "results_store.joblib":
                        model_name = item_name.replace(".joblib", "")
                        model_path = os.path.join(current_artifact_dir, item_name)
                        models_to_plot_pred.append({'name': model_name, 'path': model_path})
                
                if models_to_plot_pred:
                    num_cols_pred = 3
                    num_rows_pred = math.ceil(len(models_to_plot_pred) / num_cols_pred)
                    fig_pred, axes_pred = plt.subplots(num_rows_pred, num_cols_pred, figsize=(18, 5 * num_rows_pred), squeeze=False)
                    axes_pred_flat = axes_pred.flatten()
                    plot_idx_pred = 0

                    for model_info in models_to_plot_pred:
                        model_name = model_info['name']
                        ax = axes_pred_flat[plot_idx_pred]
                        
                        try:
                            model = joblib.load(model_info['path'])
                        except Exception as e:
                            print(f"    Error loading model {model_name} for {current_target_variable}, H{horizon_val}: {e}")
                            ax.text(0.5, 0.5, f"Error loading model:\n{model_name}", ha='center', va='center', transform=ax.transAxes, color='red')
                            ax.set_title(f"{model_name} (Load Error)", fontsize=10)
                            plot_idx_pred += 1
                            continue

                        predictions_train_raw = model.predict(X_train_scaled_df)
                        predictions_train_series = pd.Series(predictions_train_raw, index=y_train.index)
                        predictions_test_raw = model.predict(X_test_scaled_df)
                        predictions_test_series = pd.Series(predictions_test_raw, index=y_test.index)

                        train_metrics = _calculate_metrics(y_train, predictions_train_raw)
                        test_metrics = _calculate_metrics(y_test, predictions_test_raw)
                        
                        current_model_metrics = {
                            'TargetVariable': current_target_variable, 'Horizon': horizon_val, 'Model': model_name,
                            **{f"Train_{k}": v for k, v in train_metrics.items()},
                            **{f"Test_{k}": v for k, v in test_metrics.items()}
                        }
                        all_calculated_metrics.append(current_model_metrics)

                        plot_aggregated_timeseries(ax, y_train, 'Actual Train', 'blue')
                        plot_aggregated_timeseries(ax, predictions_train_series, 'Predicted Train', 'orange', line_style='--')
                        plot_aggregated_timeseries(ax, y_test, 'Actual Test', 'green')
                        plot_aggregated_timeseries(ax, predictions_test_series, 'Predicted Test', 'red', line_style='--')
                        
                        ax.set_title(f"{model_name} (Test MAPE: {test_metrics.get('MAPE', float('nan')):.2f}%)", fontsize=10)
                        ax.set_xlabel('Date', fontsize=8)
                        ax.set_ylabel(pub_target_name, fontsize=8)
                        if not y_train.empty or not y_test.empty:
                            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                        ax.tick_params(axis='x', rotation=45, labelsize=7)
                        ax.tick_params(axis='y', labelsize=7)
                        ax.legend(fontsize=7)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        plot_idx_pred += 1
                    
                    for i in range(plot_idx_pred, len(axes_pred_flat)): fig_pred.delaxes(axes_pred_flat[i])
                    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect for suptitle
                    plt.show()

                # --- Plot Feature Importances ---
                models_for_fi_this_run = [m for m in MODELS_FOR_FI_PLOT if os.path.exists(os.path.join(current_artifact_dir, f"{m}.joblib"))]
                if models_for_fi_this_run:
                    num_fi_cols = 3
                    num_fi_rows = math.ceil(len(models_for_fi_this_run) / num_fi_cols)
                    fig_fi, axes_fi = plt.subplots(num_fi_rows, num_fi_cols, figsize=(num_fi_cols * 4.5, num_fi_rows * 4), squeeze=False) # Adjusted figsize
                    axes_fi_flat = axes_fi.flatten()
                    plot_idx_fi = 0

                    for model_name_fi in models_for_fi_this_run:
                        ax_fi = axes_fi_flat[plot_idx_fi]
                        model_path_fi = os.path.join(current_artifact_dir, f"{model_name_fi}.joblib")
                        model_obj_fi = joblib.load(model_path_fi)

                        # Find the metrics for this model, target, horizon
                        r2_val_fi = np.nan
                        for m_metrics in all_calculated_metrics:
                            if m_metrics['TargetVariable'] == current_target_variable and \
                                m_metrics['Horizon'] == horizon_val and \
                                m_metrics['Model'] == model_name_fi:
                                r2_val_fi = m_metrics.get('Test_R2', np.nan) if 'Test_R2' in m_metrics else np.nan
                                break
                        
                        plot_feature_importance(
                            model_obj_fi, feature_names, ax=ax_fi,
                            top_n=TOP_N_FEATURES, r2_score_val=r2_val_fi, 
                            model_display_name=model_name_fi
                        )
                        plot_idx_fi += 1

                    for i in range(plot_idx_fi, len(axes_fi_flat)): fig_fi.delaxes(axes_fi_flat[i])
                    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect for suptitle
                    plt.show()

                # --- Display Selected Features for RFE Models ---
                print(f"    Selected Features by RFE Models (Target: {pub_target_name}, Horizon: {horizon_val}):")
                found_rfe = False
                for item_name in os.listdir(current_artifact_dir):
                    if item_name.endswith(".joblib") and "RFE" in item_name: # Simple check for RFE in name
                        model_name_rfe = item_name.replace(".joblib", "")
                        model_path_rfe = os.path.join(current_artifact_dir, item_name)
                        model_obj_rfe = joblib.load(model_path_rfe)

                        if isinstance(model_obj_rfe, Pipeline) and 'rfe' in model_obj_rfe.named_steps:
                            rfe_step = model_obj_rfe.named_steps['rfe']
                            final_estimator_name = [name for name in model_obj_rfe.named_steps if name != 'rfe'][0]
                            if hasattr(rfe_step, 'support_') and rfe_step.support_ is not None:
                                selected_features_raw = np.array(feature_names)[rfe_step.support_]
                                selected_features = [PUBLICATION_NAMES.get(name, name) for name in selected_features_raw]
                                print(f"      Model: {model_name_rfe} (Estimator: {final_estimator_name})")
                                print(f"        Selected {len(selected_features)} features: {list(selected_features)}")
                                found_rfe = True
                            else:
                                print(f"      Model: {model_name_rfe} - RFE step not fitted or 'support_' not available.")
                if not found_rfe:
                    print("      No RFE models with selection info found for this horizon.")

# --- Display Aggregated Metrics ---
if all_calculated_metrics:
    metrics_df = pd.DataFrame(all_calculated_metrics)
    metrics_df['TargetVariable'] = metrics_df['TargetVariable'].map(PUBLICATION_NAMES).fillna(metrics_df['TargetVariable'])
    # Define a more comprehensive column order
    metric_cols_ordered = ['TargetVariable', 'Horizon', 'Model', 
                            'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_MAPE', 
                            'Train_RMSE', 'Train_MAE', 'Train_R2', 'Train_MAPE',
                            'Test_MSE', 'Train_MSE']
    # Filter to only existing columns in metrics_df to avoid KeyError
    existing_metric_cols = [col for col in metric_cols_ordered if col in metrics_df.columns]
    metrics_df = metrics_df[existing_metric_cols]
    metrics_df = metrics_df.sort_values(by=['TargetVariable', 'Horizon', 'Test_RMSE'])
    
    print("\n\n--- Aggregated Performance Metrics ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        display(metrics_df)



