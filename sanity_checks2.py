import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson
from IPython.display import display
import tensorflow as tf # Needed for loading Keras models

# --- Configuration ---
ARTIFACTS_BASE_DIR = "model_run_artifacts_test3"  # Should match test3.py
DEFAULT_TARGET_VARIABLE = 'non_interest_income_to_assets' # Fallback if inference fails
# Define which horizons and models to check, e.g., [1] or load dynamically
# For now, let's try to infer from saved artifacts or check all.
HORIZONS_TO_CHECK = None # None to infer from results_store, or e.g. [1, 2]
MODELS_TO_CHECK = None   # None to check all models found, or e.g. ["XGBoost", "Ridge"]

# --- Helper Functions ---

def load_results_store(artifacts_dir):
    """Loads the results_store dictionary."""
    path = os.path.join(artifacts_dir, "results_store.joblib")
    if os.path.exists(path):
        print(f"Loading results_store from {path}")
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading results_store.joblib: {e}")
            return None
    else:
        print(f"Error: results_store.joblib not found at {path}")
        return None

def load_summary_metrics_df(artifacts_dir):
    """Loads the final_summary_metrics.csv DataFrame."""
    path = os.path.join(artifacts_dir, "final_summary_metrics.csv")
    if os.path.exists(path):
        print(f"Loading final_summary_metrics from {path}")
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Error loading final_summary_metrics.csv: {e}")
            return None
    else:
        print(f"Warning: final_summary_metrics.csv not found at {path}")
        return None

def load_model(artifacts_dir, horizon, model_name):
    """Loads a specific saved model (joblib or Keras native)."""
    joblib_path = os.path.join(artifacts_dir, f"horizon_{horizon}", f"{model_name}.joblib")
    keras_path = os.path.join(artifacts_dir, f"horizon_{horizon}", f"{model_name}_keras_model")

    if os.path.exists(joblib_path):
        try:
            print(f"  Attempting to load {model_name} (H{horizon}) from {joblib_path}")
            return joblib.load(joblib_path)
        except Exception as e:
            print(f"  Error loading model {model_name} for H{horizon} from {joblib_path}: {e}")
            return None
    elif os.path.exists(keras_path):
         # Check if it's a directory (Keras native save format)
         if os.path.isdir(keras_path):
            try:
                print(f"  Attempting to load {model_name} (H{horizon}) from Keras native format at {keras_path}")
                # Ensure TensorFlow is not too verbose during loading
                original_tf_level = tf.get_logger().level
                tf.get_logger().setLevel('ERROR')
                model = tf.keras.models.load_model(keras_path)
                tf.get_logger().setLevel(original_tf_level) # Restore original level
                # For KerasRegressor, we saved the underlying Keras model.
                # We might need to wrap it back if the plotting/prediction functions
                # expect the scikeras KerasRegressor object.
                # However, the predict method is usually available directly on the Keras model.
                # Let's return the raw Keras model for now and see if predict works.
                # If not, we might need to reconstruct the KerasRegressor wrapper.
                return model
            except Exception as e:
                print(f"  Error loading Keras model {model_name} for H{horizon} from {keras_path}: {e}")
                return None
         else:
             print(f"  Warning: Keras path exists but is not a directory for {model_name}, H{horizon} at {keras_path}")
             return None
    else:
        print(f"  Warning: Model file or directory not found for {model_name}, H{horizon} at {joblib_path} or {keras_path}")
        return None

def load_data_for_horizon(artifacts_dir, horizon):
    """Loads X_test_scaled, y_test, X_train_scaled, y_train for a given horizon."""
    data_dir = os.path.join(artifacts_dir, f"horizon_{horizon}", "data")
    data_paths = {
        "X_test_scaled": os.path.join(data_dir, "X_test_scaled.parquet"),
        "y_test": os.path.join(data_dir, "y_test.parquet"),
        "X_train_scaled": os.path.join(data_dir, "X_train_scaled.parquet"), # For train error checks
        "y_train": os.path.join(data_dir, "y_train.parquet")      # For train error checks
    }
    loaded_data = {}
    for key, path in data_paths.items():
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                if key.startswith("y_"): # y_test and y_train are saved as single-column DFs
                    loaded_data[key] = df[df.columns[0]] if not df.empty else pd.Series(dtype='float64')
                else:
                    loaded_data[key] = df
            except Exception as e:
                print(f"  Error loading {key} for H{horizon} from {path}: {e}")
                loaded_data[key] = None if not key.startswith("y_") else pd.Series(dtype='float64')
        else:
            print(f"  Warning: Data file not found for {key}, H{horizon} at {path}")
            loaded_data[key] = None if not key.startswith("y_") else pd.Series(dtype='float64')
    return loaded_data.get("X_test_scaled"), loaded_data.get("y_test"), loaded_data.get("X_train_scaled"), loaded_data.get("y_train")

def get_predictions(model, X_data):
    """Safely gets predictions from a model."""
    if model is None or X_data is None or X_data.empty:
        print("    Prediction skipped: Model or data is None/empty.")
        return np.array([])
    try:
        # Ensure X_data is in a format the model expects (e.g., numpy array for Keras)
        if isinstance(X_data, pd.DataFrame):
            X_data_predict = X_data.values # Convert DataFrame to numpy array
        else:
            X_data_predict = X_data

        predictions = model.predict(X_data_predict)

        # Handle potential shape issues with Keras predictions (e.g., (N, 1) vs (N,))
        if isinstance(predictions, np.ndarray) and predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.flatten()

        # Ensure predictions have the same index as the input data for alignment
        if isinstance(X_data, pd.DataFrame) and len(predictions) == len(X_data):
             predictions = pd.Series(predictions, index=X_data.index)

        return predictions
    except Exception as e:
        print(f"    Error during prediction: {e}")
        return np.array([])

# --- Numerical Check Functions ---

def summarize_predictions_and_errors(y_true, predictions, model_name, horizon, data_type="Test"):
    """Prints summary statistics of predictions and errors."""
    if y_true is None or y_true.empty or predictions is None or len(predictions) != len(y_true):
        print(f"  {data_type} Summary for {model_name} (H{horizon}): Insufficient data or mismatched lengths.")
        return None, None

    # Ensure predictions and y_true are aligned if they are pandas Series
    if isinstance(y_true, pd.Series) and isinstance(predictions, pd.Series):
        common_index = y_true.index.intersection(predictions.index)
        if common_index.empty:
             print(f"  {data_type} Summary for {model_name} (H{horizon}): No common index between y_true and predictions.")
             return None, None
        y_true_aligned = y_true.loc[common_index]
        predictions_aligned = predictions.loc[common_index]
    else: # Assume they are numpy arrays or already aligned
        y_true_aligned = y_true
        predictions_aligned = predictions

    if len(y_true_aligned) == 0:
         print(f"  {data_type} Summary for {model_name} (H{horizon}): Aligned data is empty.")
         return None, None

    errors = predictions_aligned - y_true_aligned

    print(f"\n--- {data_type} Data: Numerical Checks for {model_name} (Horizon {horizon}) ---")
    print(f"  Number of samples: {len(y_true_aligned)}")
    print("  Predictions Summary:")
    print(f"    Min: {np.min(predictions_aligned):.4f}, Max: {np.max(predictions_aligned):.4f}")
    print(f"    Mean: {np.mean(predictions_aligned):.4f}, Std: {np.std(predictions_aligned):.4f}")
    print(f"    NaNs: {np.isnan(predictions_aligned).sum()}, Infs: {np.isinf(predictions_aligned).sum()}")

    print("  Errors (Pred - Actual) Summary:")
    print(f"    Min: {np.min(errors):.4f}, Max: {np.max(errors):.4f}")
    print(f"    Mean Error (Bias): {np.mean(errors):.4f}, Std Error: {np.std(errors):.4f}")
    print(f"    MAE: {mean_absolute_error(y_true_aligned, predictions_aligned):.4f}")
    print(f"    RMSE: {np.sqrt(mean_squared_error(y_true_aligned, predictions_aligned)):.4f}")

    # Calculate MAPE, handling potential division by zero
    y_true_no_zeros = y_true_aligned.replace(0, np.nan).dropna() # Use np.nan for robustness
    if not y_true_no_zeros.empty:
        predictions_for_mape = pd.Series(predictions_aligned, index=y_true_aligned.index).loc[y_true_no_zeros.index]
        mape = np.mean(np.abs((y_true_no_zeros - predictions_for_mape) / y_true_no_zeros)) * 100
        print(f"    MAPE: {mape:.4f}%")
    else:
        print("    MAPE: NaN (Target variable is zero for all non-NaN values)")

    return predictions_aligned, errors

def analyze_residuals_numerical(residuals, model_name, horizon, data_type="Test"):
    """Performs numerical tests on residuals."""
    if residuals is None or len(residuals) < 8: # Shapiro needs at least 3, D-W more.
        print(f"  Residual Analysis for {model_name} (H{horizon}, {data_type}): Not enough residuals for tests.")
        return

    print(f"  Residuals Numerical Analysis ({data_type}):")
    # Normality (Shapiro-Wilk)
    if len(residuals) >=3: # Shapiro-Wilk needs at least 3 samples
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            print(f"    Shapiro-Wilk Normality Test: Statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
            if shapiro_p > 0.05:
                print("      (Residuals appear normally distributed, p > 0.05)")
            else:
                print("      (Residuals do NOT appear normally distributed, p <= 0.05)")
        except Exception as e:
            print(f"    Error during Shapiro-Wilk test: {e}")

    # Autocorrelation (Durbin-Watson) - assumes residuals are ordered by time
    # D-W test is for residuals from a regression model.
    # Values: 0-2 (positive autocorrelation), 2 (no autocorrelation), 2-4 (negative autocorrelation)
    # NOTE: Durbin-Watson is strictly applicable to time series data where observations are
    # ordered sequentially. For panel data, applying it directly to the flattened residuals
    # might not be the most appropriate test for panel-specific autocorrelation.
    # Consider panel-specific tests if this is a critical assumption.
    if len(residuals) > 15 : # D-W typically needs more samples
        try:
            dw_stat = durbin_watson(residuals)
            print(f"    Durbin-Watson Autocorrelation Test: Statistic={dw_stat:.4f}")
            if 1.5 < dw_stat < 2.5:
                print("      (Little to no first-order autocorrelation detected)")
            elif dw_stat <= 1.5:
                print("      (Potential positive first-order autocorrelation detected)")
            else: # dw_stat >= 2.5
                print("      (Potential negative first-order autocorrelation detected)")
            print("    (Note: Durbin-Watson test assumptions may not fully hold for panel data)")
        except Exception as e:
            print(f"    Error during Durbin-Watson test: {e}")
    else:
        print(f"    Durbin-Watson test skipped (need >15 residuals, have {len(residuals)})")


# --- Visual Check Functions ---

def plot_aggregated_timeseries(ax, data_series, label_prefix, color, line_style='-'):
    """Plots the mean/median and IQR of a data series aggregated by date."""
    if data_series is None or data_series.empty:
        print(f"Skipping {label_prefix} aggregated plot: data is empty or None.")
        return
    # Ensure index is a MultiIndex and contains 'date'
    if not isinstance(data_series.index, pd.MultiIndex) or 'date' not in data_series.index.names:
        print(f"Skipping {label_prefix} aggregated plot: data index is not a MultiIndex with 'date'. Index type: {type(data_series.index)}")
        return

    try:
        # Calculate quantiles and mean aggregated by date
        aggregated_data = data_series.groupby(level='date').agg(['mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.50), lambda x: x.quantile(0.75)])
        aggregated_data.columns = ['mean', 'q25', 'q50', 'q75'] # Rename columns

        if aggregated_data.empty:
            print(f"Warning: Empty aggregated data for {label_prefix}. Skipping plot.")
            return

        dates_idx = aggregated_data.index
        mean_values = aggregated_data['mean']
        median_values = aggregated_data['q50'] # Use median for the line
        q1_values = aggregated_data['q25']
        q3_values = aggregated_data['q75']

        # Plot median line
        ax.plot(dates_idx, median_values, label=f'{label_prefix} Median', color=color, linestyle=line_style)
        # Plot IQR band
        ax.fill_between(dates_idx, q1_values, q3_values, color=color, alpha=0.2, label=f'{label_prefix} IQR')

    except Exception as e:
        print(f"Error generating aggregated timeseries plot for {label_prefix}: {e}")
        # Optionally plot individual lines if aggregation fails
        # sns.lineplot(data=data_series.reset_index(), x='date', y=data_series.name, hue='id', ax=ax, legend=False, alpha=0.5)
        # ax.set_title(f'{label_prefix} (Individual Lines - Aggregation Failed)')


def plot_scatter_actual_vs_predicted(y_true, predictions, model_name, horizon, ax, target_variable_name, data_type="Test"):
    """Plots actual vs. predicted values as a scatter plot."""
    if y_true is None or y_true.empty or predictions is None or len(predictions) != len(y_true):
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nScatter Actual vs. Predicted (Data N/A)")
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
        return

    try:
        # Ensure y_true and predictions are aligned and are numpy arrays for plotting
        if isinstance(y_true, pd.Series) and isinstance(predictions, pd.Series):
             common_index = y_true.index.intersection(predictions.index)
             y_true_plot = y_true.loc[common_index].values
             predictions_plot = predictions.loc[common_index].values
        else:
             y_true_plot = y_true
             predictions_plot = predictions

        if len(y_true_plot) == 0:
             ax.set_title(f"{model_name} (H{horizon}, {data_type})\nScatter Actual vs. Predicted (Data N/A)")
             ax.text(0.5, 0.5, "Aligned data is empty", ha='center', va='center', transform=ax.transAxes)
             return

        min_val = min(y_true_plot.min(), predictions_plot.min())
        max_val = max(y_true_plot.max(), predictions_plot.max())

        ax.scatter(y_true_plot, predictions_plot, alpha=0.5, label=f"{data_type} Data", s=5) # Reduced marker size
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nActual vs. Predicted (Scatter)")
        ax.set_xlabel(f"Actual {target_variable_name}")
        ax.set_ylabel(f"Predicted {target_variable_name}")
        ax.legend()
        ax.grid(True)
    except Exception as e:
        print(f"Error generating scatter plot for {model_name} H{horizon} ({data_type}): {e}")
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nScatter Plot (Error)")
        ax.text(0.5, 0.5, "Error generating plot", ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()


def plot_residuals_vs_predicted(residuals, predictions, model_name, horizon, ax, data_type="Test"):
    """Plots residuals vs. predicted values."""
    if residuals is None or predictions is None or len(predictions) != len(residuals):
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals vs. Predicted (Data N/A)")
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
        return

    try:
        # Ensure residuals and predictions are aligned and are numpy arrays for plotting
        if isinstance(residuals, pd.Series) and isinstance(predictions, pd.Series):
             common_index = residuals.index.intersection(predictions.index)
             residuals_plot = residuals.loc[common_index].values
             predictions_plot = predictions.loc[common_index].values
        else:
             residuals_plot = residuals
             predictions_plot = predictions

        if len(residuals_plot) == 0:
             ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals vs. Predicted (Data N/A)")
             ax.text(0.5, 0.5, "Aligned data is empty", ha='center', va='center', transform=ax.transAxes)
             return

        ax.scatter(predictions_plot, residuals_plot, alpha=0.5, s=5) # Reduced marker size
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals vs. Predicted")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals (Pred - Actual)")
        ax.grid(True)
    except Exception as e:
        print(f"Error generating residuals vs. predicted plot for {model_name} H{horizon} ({data_type}): {e}")
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals vs. Predicted (Error)")
        ax.text(0.5, 0.5, "Error generating plot", ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()


def plot_residuals_histogram(residuals, model_name, horizon, ax, data_type="Test"):
    """Plots a histogram of residuals."""
    if residuals is None or len(residuals) == 0:
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals Histogram (Data N/A)")
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
        return

    try:
        # Ensure residuals are a numpy array for plotting
        residuals_plot = residuals.values if isinstance(residuals, pd.Series) else residuals
        sns.histplot(residuals_plot, kde=True, ax=ax)
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals Distribution")
        ax.set_xlabel("Residuals (Pred - Actual)")
        ax.set_ylabel("Frequency")
        ax.grid(True)
    except Exception as e:
        print(f"Error generating residuals histogram for {model_name} H{horizon} ({data_type}): {e}")
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals Histogram (Error)")
        ax.text(0.5, 0.5, "Error generating plot", ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()

def plot_residuals_qq(residuals, model_name, horizon, ax, data_type="Test"):
    """Plots a Q-Q plot of residuals."""
    if residuals is None or len(residuals) < 2 : # qqplot needs at least 2 points
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals Q-Q Plot (Data N/A)")
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
        return
    try:
        # Ensure residuals are a numpy array for plotting
        residuals_plot = residuals.values if isinstance(residuals, pd.Series) else residuals
        qqplot(residuals_plot, line='s', ax=ax) # 's' for standardized line
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals Q-Q Plot")
        # Adjust appearance if needed
        if ax.get_lines():
            ax.get_lines()[0].set_markerfacecolor('blue') # Points
            ax.get_lines()[0].set_markeredgecolor('blue')
            ax.get_lines()[0].set_markersize(3.0)
            if len(ax.get_lines()) > 1:
                 ax.get_lines()[1].set_color('red') # Reference line
    except Exception as e:
        print(f"Error in Q-Q plot for {model_name} H{horizon} ({data_type}): {e}")
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals Q-Q Plot (Error)")
        ax.text(0.5, 0.5, "Error generating plot", ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()


def plot_feature_importance(model, feature_names, model_name, horizon, ax, top_n=15):
    """Plots feature importance or coefficients."""
    if model is None or not feature_names:
        ax.set_title(f"{model_name} (H{horizon})\nFeature Importance (N/A)")
        ax.text(0.5, 0.5, "Model or features not available", ha='center', va='center', transform=ax.transAxes)
        return

    importances = None
    # Handle different model types
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        plot_title = 'Feature Importances'
    elif hasattr(model, 'coef_'):
        importances = model.coef_
        plot_title = 'Feature Coefficients'
        if len(importances.shape) > 1: # E.g. some multi-output or complex models
            # For simplicity, take the mean absolute coefficient across outputs if multi-output
            importances = np.mean(np.abs(importances), axis=0)
            plot_title = 'Mean Absolute Feature Coefficients'
    elif hasattr(model, 'model_') and hasattr(model.model_, 'coef_'): # Check scikeras wrapper for coef_
         importances = model.model_.coef_
         plot_title = 'Feature Coefficients'
         if len(importances.shape) > 1:
             importances = np.mean(np.abs(importances), axis=0)
             plot_title = 'Mean Absolute Feature Coefficients'
    # Add more model types here if needed (e.g., specific deep learning layer weights, though less common for direct feature importance)

    if importances is None or len(importances) != len(feature_names):
        print(f"  Feature importance/coefficients not available or mismatch with feature names for {model_name} (H{horizon}).")
        ax.set_title(f"{model_name} (H{horizon})\nFeature Importance (Not available)")
        ax.text(0.5, 0.5, "Importance/coeffs not found or mismatch", ha='center', va='center', transform=ax.transAxes)
        return

    # Sort by absolute importance/coefficient value
    indices = np.argsort(np.abs(importances))[::-1]
    top_indices = indices[:top_n]

    # Use the actual importance/coefficient values for plotting the bars
    top_values = importances[top_indices]
    top_feature_names = np.array(feature_names)[top_indices]

    ax.bar(range(len(top_values)), top_values, align='center')
    ax.set_xticks(range(len(top_values)))
    ax.set_xticklabels(top_feature_names, rotation=45, ha="right")
    ax.set_title(f"{model_name} (H{horizon})\nTop {top_n} {plot_title}")
    ax.set_ylabel("Value")
    plt.tight_layout()

# --- Helper for Target Variable Inference ---
def infer_target_variable_from_y_name(y_series_name: str):
    """
    Infers the base target variable name from a y_series name like 'base_target_target_h1'.
    Returns 'base_target' or None if parsing fails.
    """
    if not y_series_name or not isinstance(y_series_name, str):
        return None

    # Split by '_target_h' and take the first part
    parts = y_series_name.split("_target_h")
    if len(parts) > 1:
        # Check if the part after _target_h is a number (the horizon)
        # This makes the parsing more robust.
        horizon_part = parts[-1] # Take the last part after the split
        # Check if it's just a number or number + _train/_test
        potential_horizon_str = horizon_part.split('_')[0]
        if potential_horizon_str.isdigit():
            return parts[0] # The base target variable name

    print(f"Warning: Could not parse base target variable from y_series_name: '{y_series_name}'")
    return None

# --- Main Execution Logic ---
def run_sanity_checks():
    print("--- Starting Sanity Checks ---")
    results_store = load_results_store(ARTIFACTS_BASE_DIR)
    summary_df = load_summary_metrics_df(ARTIFACTS_BASE_DIR)

    if results_store is None:
        print("Cannot proceed without results_store. Exiting.")
        return

    if summary_df is not None:
        print("\n--- Overall Performance Summary (from 4_regressions.py) ---")
        display(summary_df)

        # Compare to DummyRegressor
        dummy_rmses = summary_df[summary_df['Model'] == 'DummyRegressor'][['Horizon', 'RMSE']].set_index('Horizon')
        if not dummy_rmses.empty:
            print("\n--- RMSE Comparison to DummyRegressor ---")
            comparison_list = []
            for idx, row in summary_df.iterrows():
                if row['Model'] != 'DummyRegressor':
                    dummy_rmse = dummy_rmses.loc[row['Horizon'], 'RMSE'] if row['Horizon'] in dummy_rmses.index else np.nan
                    # Calculate improvement: positive means the model is better than dummy
                    improvement = dummy_rmse - row['RMSE'] if not np.isnan(dummy_rmse) and not np.isnan(row['RMSE']) else np.nan
                    comparison_list.append({
                        'Horizon': row['Horizon'],
                        'Model': row['Model'],
                        'Model_RMSE': row['RMSE'],
                        'Dummy_RMSE': dummy_rmse,
                        'Improvement_vs_Dummy': improvement # Positive is good
                    })
            comparison_df = pd.DataFrame(comparison_list).sort_values(by=['Horizon', 'Model_RMSE'])
            display(comparison_df)
        else:
            print("\nDummyRegressor results not found in summary for comparison.")


    horizons_in_results = sorted(results_store.keys())
    horizons_to_run = HORIZONS_TO_CHECK if HORIZONS_TO_CHECK else horizons_in_results

    effective_target_variable = DEFAULT_TARGET_VARIABLE # Start with default

    # Try to infer target variable from the first available horizon's y_test data
    inferred_successfully = False
    if horizons_to_run:
        for h_check in horizons_to_run:
            if h_check in results_store:
                _, y_test_for_infer, _, y_train_for_infer = load_data_for_horizon(ARTIFACTS_BASE_DIR, h_check)
                y_series_for_name = None
                if y_test_for_infer is not None and not y_test_for_infer.empty and y_test_for_infer.name:
                    y_series_for_name = y_test_for_infer
                elif y_train_for_infer is not None and not y_train_for_infer.empty and y_train_for_infer.name:
                    y_series_for_name = y_train_for_infer

                if y_series_for_name is not None:
                    parsed_name = infer_target_variable_from_y_name(y_series_for_name.name)
                    if parsed_name:
                        effective_target_variable = parsed_name
                        inferred_successfully = True
                        print(f"\nSuccessfully inferred target variable as: '{effective_target_variable}'")
                        break # Stop after first successful inference
    if not inferred_successfully:
        print(f"\nCould not infer target variable from y_test/y_train names. Using default: '{DEFAULT_TARGET_VARIABLE}'")

    for horizon in horizons_to_run:
        if horizon not in results_store:
            print(f"\n--- Horizon {horizon}: Not found in results_store. Skipping. ---")
            continue

        print(f"\n\n--- Processing Horizon: {horizon} ---")
        X_test_scaled, y_test, X_train_scaled, y_train = load_data_for_horizon(ARTIFACTS_BASE_DIR, horizon)

        if y_test is None or y_test.empty:
            print(f"  No y_test data available for horizon {horizon}. Skipping detailed checks for this horizon.")
            # Still attempt to load models and check train data if available

        models_in_horizon = results_store[horizon].keys()
        models_to_run_for_horizon = MODELS_TO_CHECK if MODELS_TO_CHECK else models_in_horizon

        for model_name in models_to_run_for_horizon:
            # Skip ensemble for individual checks if base models are checked
            if model_name == "VotingEnsemble":
                 print(f"\n  --- Skipping individual checks for {model_name} (H{horizon}) ---")
                 continue # Skip to the next model

            print(f"\n  --- Model: {model_name} (Horizon {horizon}) ---")

            # Check if model training failed based on results_store entry
            model_entry = results_store[horizon].get(model_name)
            if model_entry is None or not isinstance(model_entry, dict):
                 print(f"    Model {model_name} (H{horizon}) entry not found or invalid in results_store. Skipping.")
                 continue

            model = load_model(ARTIFACTS_BASE_DIR, horizon, model_name)

            if model is None:
                print(f"    Could not load model {model_name} (H{horizon}). Skipping checks.")
                # Check if the model_object was None in results_store (indicating training/save failure)
                if model_entry.get('model_object') is None:
                     print(f"    Red Flag: Model object for {model_name} (H{horizon}) was None in results_store (training/save likely failed).")
                continue

            # --- Test Data Checks ---
            if X_test_scaled is not None and not X_test_scaled.empty and y_test is not None and not y_test.empty:
                predictions_test = get_predictions(model, X_test_scaled)
                if predictions_test is not None and len(predictions_test) > 0:
                    predictions_test_series = pd.Series(predictions_test, index=X_test_scaled.index) # Ensure Series with index for plotting/alignment
                    _, residuals_test = summarize_predictions_and_errors(y_test, predictions_test_series, model_name, horizon, "Test")
                    if residuals_test is not None:
                         analyze_residuals_numerical(residuals_test, model_name, horizon, "Test")

                    # Visualizations for Test Data
                    # Determine number of plots: 4 standard residual plots + 1 scatter + 1 aggregated time series + 1 feature importance (if applicable)
                    num_plots_base = 6 # Scatter, ResVSPred, ResHist, ResQQ, Aggregated TS, Feature Importance
                    has_feature_importance = hasattr(model, 'feature_importances_') or hasattr(model, 'coef_') or (hasattr(model, 'model_') and hasattr(model.model_, 'coef_'))
                    num_plots = num_plots_base # We'll arrange them in a grid

                    # Create figure for residual/scatter plots (4+1 = 5 plots)
                    fig1, axes1 = plt.subplots(1, 5, figsize=(5 * 5, 4))
                    fig1.suptitle(f"Test Data Diagnostics: {model_name} - Horizon {horizon}", fontsize=16, y=1.03)

                    plot_scatter_actual_vs_predicted(y_test, predictions_test_series, model_name, horizon, axes1[0], effective_target_variable, "Test")
                    if residuals_test is not None:
                        plot_residuals_vs_predicted(residuals_test, predictions_test_series, model_name, horizon, axes1[1], "Test")
                        plot_residuals_histogram(residuals_test, model_name, horizon, axes1[2], "Test")
                        plot_residuals_qq(residuals_test, model_name, horizon, axes1[3], "Test")
                    # Add a blank subplot or another plot type if needed to fill the grid
                    # For now, leave axes1[4] empty or use it for something else
                    axes1[4].set_visible(False) # Hide the unused 5th subplot in the first row

                    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
                    plt.show()

                    # Create figure for aggregated time series plot
                    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
                    fig2.suptitle(f"Test Data: Aggregated Actual vs. Predicted - {model_name} (Horizon {horizon})", fontsize=16, y=1.03)
                    plot_aggregated_timeseries(ax2, y_test, 'Actual Test', 'green')
                    plot_aggregated_timeseries(ax2, predictions_test_series, 'Predicted Test', 'red', line_style='--')
                    ax2.set_title(f'Aggregated Actual vs. Predicted (Median and IQR)')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel(effective_target_variable)
                    ax2.legend()
                    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
                    # Format x-axis as dates if possible
                    if isinstance(y_test.index, pd.MultiIndex) and 'date' in y_test.index.names:
                         try:
                             ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                             fig2.autofmt_xdate() # Auto-rotate date labels
                         except Exception as date_format_e:
                             print(f"Warning: Could not format x-axis as dates: {date_format_e}")
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.show()

                    # Feature Importance Plot (if applicable)
                    if has_feature_importance and X_test_scaled is not None and not X_test_scaled.empty:
                        fig_fi, ax_fi = plt.subplots(1, 1, figsize=(10, 6))
                        plot_feature_importance(model, X_test_scaled.columns.tolist(), model_name, horizon, ax_fi)
                        plt.show()
                    elif has_feature_importance:
                         print(f"    Skipping Feature Importance plot for {model_name} (H{horizon}): X_test_scaled is empty.")

                else:
                    print(f"    Could not generate test predictions for {model_name} (H{horizon}).")
            else:
                print(f"    Test data (X_test_scaled or y_test) not available for {model_name}, H{horizon}. Skipping test data checks.")
                # Check if test metrics were NaN in results_store
                if np.isnan(model_entry.get('RMSE', np.nan)):
                     print(f"    Red Flag: Test metrics for {model_name} (H{horizon}) were NaN in results_store (evaluation likely failed).")

            # --- Train Data Checks (Optional, good for overfitting check) ---
            if X_train_scaled is not None and not X_train_scaled.empty and y_train is not None and not y_train.empty:
                print(f"\n    --- Train Data Checks for {model_name} (Horizon {horizon}) ---")
                predictions_train = get_predictions(model, X_train_scaled)
                if predictions_train is not None and len(predictions_train) > 0:
                    predictions_train_series = pd.Series(predictions_train, index=X_train_scaled.index) # Ensure Series with index
                    _, residuals_train = summarize_predictions_and_errors(y_train, predictions_train_series, model_name, horizon, "Train")
                    if residuals_train is not None:
                        analyze_residuals_numerical(residuals_train, model_name, horizon, "Train")
                        # Optionally, add visual checks for train data too (e.g., residual plots)

                    # Overfitting Check
                    test_rmse = model_entry.get('RMSE', np.nan)
                    train_rmse = model_entry.get('RMSE_train', np.nan)
                    if not np.isnan(test_rmse) and not np.isnan(train_rmse):
                        print(f"    Overfitting Check: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}")
                        # Simple heuristic: if test RMSE is significantly higher than train RMSE
                        if test_rmse > train_rmse * 1.5: # Threshold can be adjusted
                             print(f"    Red Flag: Potential overfitting detected (Test RMSE is significantly higher than Train RMSE).")
                        elif test_rmse < train_rmse * 0.8: # Sometimes test RMSE can be lower, less common but worth noting
                             print(f"    Note: Test RMSE is lower than Train RMSE.")

                else:
                    print(f"    Could not generate train predictions for {model_name} (H{horizon}).")
            else:
                print(f"    Train data (X_train_scaled or y_train) not available for {model_name}, H{horizon}. Skipping train data checks.")

    # --- Voting Ensemble Checks (Optional, can be added here if needed) ---
    # You could add a separate loop or section here to specifically check the VotingEnsemble
    # if you want to see its predictions/residuals, similar to the individual models.
    # This would involve loading the VotingEnsemble model and using the test data.
    # For brevity, I'm skipping detailed plots for the ensemble in this initial script,
    # but its overall metrics are in the summary table.
    print(f"\n--- Individual Model Checks for Horizon {horizon} Completed ---")


if __name__ == "__main__":
    run_sanity_checks()
    print("\n--- Sanity Checks Completed ---")