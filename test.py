import pandas as pd
import numpy as np
import joblib
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns # For plot_aggregated_timeseries if it uses seaborn features
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from IPython.display import display

# --- Configuration ---
ARTIFACTS_BASE_DIR = "model_run_artifacts_test3"  # Should match your main script

# --- Discover Targets, Horizons, and Models from Artifacts ---
discovered_targets = []
discovered_horizons_map = {}  # Maps target_name to a list of horizons
discovered_models_map = {}    # Maps (target_name, horizon) to a list of models

# Discover targets, horizons, and models from the artifacts directory
if os.path.exists(ARTIFACTS_BASE_DIR):
    for target_dir_name in os.listdir(ARTIFACTS_BASE_DIR):
        target_path = os.path.join(ARTIFACTS_BASE_DIR, target_dir_name)
        if target_dir_name.startswith("target_") and os.path.isdir(target_path):
            # Assumes original target names don't contain '/' or '\', so sanitization is direct
            target_name = target_dir_name[len("target_"):]
            discovered_targets.append(target_name)
            discovered_horizons_map[target_name] = []

            for horizon_dir_name in os.listdir(target_path):
                horizon_path = os.path.join(target_path, horizon_dir_name)
                if horizon_dir_name.startswith("horizon_") and os.path.isdir(horizon_path):
                    try:
                        horizon = int(horizon_dir_name[len("horizon_"):])
                        discovered_horizons_map[target_name].append(horizon)
                        discovered_models_map[(target_name, horizon)] = []

                        data_and_model_path = os.path.join(horizon_path) # Path to horizon_X folder
                        for item_name in os.listdir(data_and_model_path):
                            item_full_path = os.path.join(data_and_model_path, item_name)
                            # Discover .joblib models (excluding data folder content)
                            if os.path.isfile(item_full_path) and item_name.endswith(".joblib"):
                                model_name_candidate = item_name[:-len(".joblib")]
                                # Add any other special joblib files to exclude here if necessary
                                if model_name_candidate not in ["results_store"]:
                                    discovered_models_map[(target_name, horizon)].append(model_name_candidate)
                    except ValueError:
                        # If horizon_X part is not an int
                        continue
    
    discovered_targets = sorted(list(set(discovered_targets)))
    for t_name in discovered_targets:
        discovered_horizons_map[t_name] = sorted(list(set(discovered_horizons_map[t_name])))
    for th_pair in discovered_models_map:
        discovered_models_map[th_pair] = sorted(list(set(discovered_models_map[th_pair])))
else:
    print(f"Warning: Artifacts directory '{ARTIFACTS_BASE_DIR}' not found. No models to debug.")

# --- Helper Functions ---

def load_model_for_debug(artifacts_dir, target_name, horizon, model_name):
    """Loads a specific saved model."""
    sanitized_target = target_name.replace('/', '_').replace('\\', '_')
    model_path = os.path.join(artifacts_dir, f"target_{sanitized_target}", f"horizon_{horizon}", f"{model_name}.joblib")
    print(f"  Loading model: {model_path}")
    return joblib.load(model_path)

def load_train_test_data_for_debug(artifacts_dir, target_name, horizon):
    """Loads X_train_scaled, y_train, X_test_scaled, and y_test for a given target and horizon."""
    sanitized_target = target_name.replace('/', '_').replace('\\', '_')
    data_dir = os.path.join(artifacts_dir, f"target_{sanitized_target}", f"horizon_{horizon}", "data")
    
    data_paths = {
        "X_train": os.path.join(data_dir, "X_train_scaled.parquet"),
        "y_train": os.path.join(data_dir, "y_train.parquet"),
        "X_test": os.path.join(data_dir, "X_test_scaled.parquet"),
        "y_test": os.path.join(data_dir, "y_test.parquet")
    }
    
    loaded_data = {}
    for name, path in data_paths.items():
        df = pd.read_parquet(path)
        if name.startswith("y_"):
            loaded_data[name] = df[df.columns[0]]
        else:
            loaded_data[name] = df
            
    return loaded_data["X_train"], loaded_data["y_train"], loaded_data["X_test"], loaded_data["y_test"]

def _calculate_metrics(y_true: pd.Series, predictions: np.ndarray, prefix: str = "") -> dict:
    """Calculates standard regression metrics."""
    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions)

    y_true_no_zeros = y_true.replace(0, np.nan).dropna()
    mape = np.nan
    if not y_true_no_zeros.empty:
        predictions_aligned = pd.Series(predictions, index=y_true.index).loc[y_true_no_zeros.index]
        mape = np.mean(np.abs((y_true_no_zeros - predictions_aligned) / y_true_no_zeros)) * 100
    
    return {
        f'{prefix}MAE': mae, f'{prefix}MSE': mse, f'{prefix}RMSE': rmse,
        f'{prefix}R2': r2, f'{prefix}MAPE': mape
    }

def plot_aggregated_timeseries(ax, data_series, label_prefix, color, line_style='-'):
    """Plots aggregated (median and IQR) time series."""
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
    median_values = quantiles_df[0.50] # Using median for the line
    q1_values = quantiles_df[0.25]
    q3_values = quantiles_df[0.75]
    
    ax.plot(dates_idx, median_values, label=f'{label_prefix} Median', color=color, linestyle=line_style)
    ax.fill_between(dates_idx, q1_values, q3_values, color=color, alpha=0.2, label=f'{label_prefix} IQR')


# --- Main Debugging Logic ---
print("--- Starting Model Prediction Debug ---")

all_metrics_data = []

for target_var in discovered_targets:
    print(f"\nInspecting Target Variable: {target_var}")
    horizons_for_target = discovered_horizons_map.get(target_var, [])
    if not horizons_for_target:
        print(f"  No horizons found for target {target_var}. Skipping.")
        continue
    for horizon in horizons_for_target:
        print(f"  Inspecting Horizon: {horizon}")
        
        X_train, y_train_actual, X_test, y_test_actual = load_train_test_data_for_debug(ARTIFACTS_BASE_DIR, target_var, horizon)
        
        models_for_th = discovered_models_map.get((target_var, horizon), [])
        if not models_for_th:
            print(f"    No models found for target {target_var}, horizon {horizon}. Skipping.")
            continue
            
        # Prepare for plotting
        num_models_to_plot = len(models_for_th)
        num_cols_plot = min(3, num_models_to_plot)
        num_rows_plot = math.ceil(num_models_to_plot / num_cols_plot)
        
        fig, axes = plt.subplots(num_rows_plot, num_cols_plot, figsize=(6 * num_cols_plot, 5 * num_rows_plot), squeeze=False)
        axes_flat = axes.flatten()
        plot_idx = 0

        for model_name in models_for_th:
            print(f"    Inspecting Model: {model_name}")
            model = load_model_for_debug(ARTIFACTS_BASE_DIR, target_var, horizon, model_name)
            
            current_metrics = {"Target": target_var, "Horizon": horizon, "Model": model_name}

            # --- Train Predictions and Metrics ---
            # Use X_train and X_test directly without reconciliation
            predictions_train = model.predict(X_train)
            predictions_train_series = pd.Series(predictions_train, index=y_train_actual.index)
            train_metrics = _calculate_metrics(y_train_actual, predictions_train, prefix="Train_")
            current_metrics.update(train_metrics)
            
            predictions_test = model.predict(X_test)
            predictions_test_series = pd.Series(predictions_test, index=y_test_actual.index)
            test_metrics = _calculate_metrics(y_test_actual, predictions_test, prefix="Test_")
            current_metrics.update(test_metrics)
            
            all_metrics_data.append(current_metrics)
            
            print(f"      Test RMSE: {current_metrics.get('Test_RMSE', np.nan):.4f}, Test R2: {current_metrics.get('Test_R2', np.nan):.4f}")

            # --- Plotting for current model ---
            if plot_idx < len(axes_flat):
                ax = axes_flat[plot_idx]
                plot_aggregated_timeseries(ax, y_train_actual, 'Actual Train', 'blue')
                plot_aggregated_timeseries(ax, predictions_train_series, 'Predicted Train', 'orange', line_style='--')
                plot_aggregated_timeseries(ax, y_test_actual, 'Actual Test', 'green')
                plot_aggregated_timeseries(ax, predictions_test_series, 'Predicted Test', 'red', line_style='--')
                
                ax.set_title(f'{model_name}\n(Test R²: {current_metrics.get("Test_R2", np.nan):.2f})', fontsize=10)
                ax.set_xlabel('Date', fontsize=8)
                ax.set_ylabel(target_var, fontsize=8)
                if (y_train_actual is not None and not y_train_actual.empty) or \
                   (y_test_actual is not None and not y_test_actual.empty):
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                ax.tick_params(axis='x', rotation=45, labelsize=7)
                ax.tick_params(axis='y', labelsize=7)
                ax.legend(fontsize=7)
                ax.grid(True, linestyle='--', alpha=0.7)
                plot_idx += 1
        
        # Clean up unused subplots for the current target/horizon figure
        for i in range(plot_idx, len(axes_flat)):
            fig.delaxes(axes_flat[i])
        
        fig.suptitle(f"Model Predictions vs Actuals\nTarget: {target_var}, Horizon: {horizon}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect for suptitle
        plt.show()

# --- Display Metrics Table ---
print("\n\n--- Performance Metrics Summary ---")
metrics_df = pd.DataFrame(all_metrics_data)
metrics_df = metrics_df.sort_values(by=["Target", "Horizon", "Test_RMSE"])
display(metrics_df)

print("\n--- Debug Script Finished ---")
