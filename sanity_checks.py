# sanity_checks.py

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
        return joblib.load(path)
    else:
        print(f"Error: results_store.joblib not found at {path}")
        return None

def load_summary_metrics_df(artifacts_dir):
    """Loads the final_summary_metrics.csv DataFrame."""
    path = os.path.join(artifacts_dir, "final_summary_metrics.csv")
    if os.path.exists(path):
        print(f"Loading final_summary_metrics from {path}")
        return pd.read_csv(path)
    else:
        print(f"Warning: final_summary_metrics.csv not found at {path}")
        return None

def load_model(artifacts_dir, horizon, model_name):
    """Loads a specific saved model."""
    path = os.path.join(artifacts_dir, f"horizon_{horizon}", f"{model_name}.joblib")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading model {model_name} for H{horizon} from {path}: {e}")
            return None
    else:
        print(f"Warning: Model file not found for {model_name}, H{horizon} at {path}")
        return None

def load_data_for_horizon(artifacts_dir, horizon):
    """Loads X_test_scaled and y_test for a given horizon."""
    data_dir = os.path.join(artifacts_dir, f"horizon_{horizon}", "data")
    data_paths = {
        "X_test": os.path.join(data_dir, "X_test_scaled.parquet"),
        "y_test": os.path.join(data_dir, "y_test.parquet"),
        "X_train": os.path.join(data_dir, "X_train_scaled.parquet"), # For train error checks
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
                print(f"Error loading {key} for H{horizon} from {path}: {e}")
                loaded_data[key] = None if not key.startswith("y_") else pd.Series(dtype='float64')
        else:
            print(f"Warning: Data file not found for {key}, H{horizon} at {path}")
            loaded_data[key] = None if not key.startswith("y_") else pd.Series(dtype='float64')
    return loaded_data.get("X_test"), loaded_data.get("y_test"), loaded_data.get("X_train"), loaded_data.get("y_train")

def get_predictions(model, X_data):
    """Safely gets predictions from a model."""
    if model is None or X_data is None or X_data.empty:
        return np.array([])
    try:
        return model.predict(X_data)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return np.array([])

# --- Numerical Check Functions ---

def summarize_predictions_and_errors(y_true, predictions, model_name, horizon, data_type="Test"):
    """Prints summary statistics of predictions and errors."""
    if y_true.empty or len(predictions) != len(y_true):
        print(f"  {data_type} Summary for {model_name} (H{horizon}): Insufficient data or mismatched lengths.")
        return None, None

    errors = predictions - y_true
    print(f"\n--- {data_type} Data: Numerical Checks for {model_name} (Horizon {horizon}) ---")
    print("  Predictions Summary:")
    print(f"    Min: {np.min(predictions):.4f}, Max: {np.max(predictions):.4f}")
    print(f"    Mean: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}")
    print(f"    NaNs: {np.isnan(predictions).sum()}, Infs: {np.isinf(predictions).sum()}")

    print("  Errors (Pred - Actual) Summary:")
    print(f"    Min: {np.min(errors):.4f}, Max: {np.max(errors):.4f}")
    print(f"    Mean Error (Bias): {np.mean(errors):.4f}, Std Error: {np.std(errors):.4f}")
    print(f"    MAE: {mean_absolute_error(y_true, predictions):.4f}")
    print(f"    RMSE: {np.sqrt(mean_squared_error(y_true, predictions)):.4f}")
    return predictions, errors

def analyze_residuals_numerical(residuals, model_name, horizon, data_type="Test"):
    """Performs numerical tests on residuals."""
    if residuals is None or len(residuals) < 8: # Shapiro needs at least 3, D-W more.
        print(f"  Residual Analysis for {model_name} (H{horizon}, {data_type}): Not enough residuals for tests.")
        return

    print(f"  Residuals Numerical Analysis ({data_type}):")
    # Normality (Shapiro-Wilk)
    if len(residuals) >=3: # Shapiro-Wilk needs at least 3 samples
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"    Shapiro-Wilk Normality Test: Statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
        if shapiro_p > 0.05:
            print("      (Residuals appear normally distributed, p > 0.05)")
        else:
            print("      (Residuals do NOT appear normally distributed, p <= 0.05)")

    # Autocorrelation (Durbin-Watson) - assumes residuals are ordered by time
    # D-W test is for residuals from a regression model.
    # Values: 0-2 (positive autocorrelation), 2 (no autocorrelation), 2-4 (negative autocorrelation)
    if len(residuals) > 15 : # D-W typically needs more samples
        dw_stat = durbin_watson(residuals)
        print(f"    Durbin-Watson Autocorrelation Test: Statistic={dw_stat:.4f}")
        if 1.5 < dw_stat < 2.5:
            print("      (Little to no first-order autocorrelation detected)")
        elif dw_stat <= 1.5:
            print("      (Potential positive first-order autocorrelation detected)")
        else: # dw_stat >= 2.5
            print("      (Potential negative first-order autocorrelation detected)")
    else:
        print(f"    Durbin-Watson test skipped (need >15 residuals, have {len(residuals)})")


# --- Visual Check Functions ---

def plot_actual_vs_predicted(y_true, predictions, dates, model_name, horizon, ax, target_variable_name, data_type="Test"):
    """Plots actual vs. predicted values over time and as a scatter plot."""
    if y_true.empty or len(predictions) != len(y_true):
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nActual vs. Predicted (Data N/A)")
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
        return

    # Time series plot
    ax.plot(dates, y_true, label='Actual', marker='.', linestyle='-', alpha=0.7)
    ax.plot(dates, predictions, label='Predicted', marker='.', linestyle='--', alpha=0.7)
    ax.set_title(f"{model_name} (H{horizon}, {data_type})\nActual vs. Predicted (Time Series)")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_variable_name)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()


def plot_scatter_actual_vs_predicted(y_true, predictions, model_name, horizon, ax, target_variable_name, data_type="Test"):
    if y_true.empty or len(predictions) != len(y_true):
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nScatter Actual vs. Predicted (Data N/A)")
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
        return

    # Scatter plot
    min_val = min(y_true.min(), predictions.min())
    max_val = max(y_true.max(), predictions.max())
    ax.scatter(y_true, predictions, alpha=0.5, label=f"{data_type} Data")
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    ax.set_title(f"{model_name} (H{horizon}, {data_type})\nActual vs. Predicted (Scatter)")
    ax.set_xlabel(f"Actual {target_variable_name}")
    ax.set_ylabel(f"Predicted {target_variable_name}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()


def plot_residuals_vs_predicted(residuals, predictions, model_name, horizon, ax, data_type="Test"):
    """Plots residuals vs. predicted values."""
    if residuals is None or len(predictions) != len(residuals):
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals vs. Predicted (Data N/A)")
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
        return

    ax.scatter(predictions, residuals, alpha=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals vs. Predicted")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Pred - Actual)")
    ax.grid(True)
    plt.tight_layout()


def plot_residuals_histogram(residuals, model_name, horizon, ax, data_type="Test"):
    """Plots a histogram of residuals."""
    if residuals is None:
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals Histogram (Data N/A)")
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
        return

    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals Distribution")
    ax.set_xlabel("Residuals (Pred - Actual)")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    plt.tight_layout()

def plot_residuals_qq(residuals, model_name, horizon, ax, data_type="Test"):
    """Plots a Q-Q plot of residuals."""
    if residuals is None or len(residuals) < 2 : # qqplot needs at least 2 points
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals Q-Q Plot (Data N/A)")
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
        return
    try:
        qqplot(residuals, line='s', ax=ax) # 's' for standardized line
        ax.set_title(f"{model_name} (H{horizon}, {data_type})\nResiduals Q-Q Plot")
        ax.get_lines()[0].set_markerfacecolor('blue') # Points
        ax.get_lines()[0].set_markeredgecolor('blue')
        ax.get_lines()[0].set_markersize(3.0)
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
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_
        if len(importances.shape) > 1: # E.g. some multi-output or complex models
            importances = importances[0] # Take the first set or average/sum as appropriate

    if importances is None:
        ax.set_title(f"{model_name} (H{horizon})\nFeature Importance (Not available for this model type)")
        ax.text(0.5, 0.5, "Importance/coeffs not found", ha='center', va='center', transform=ax.transAxes)
        return

    indices = np.argsort(np.abs(importances))[::-1]
    top_indices = indices[:top_n]

    ax.bar(range(len(top_indices)), importances[top_indices], align='center')
    ax.set_xticks(range(len(top_indices)))
    ax.set_xticklabels(np.array(feature_names)[top_indices], rotation=45, ha="right")
    ax.set_title(f"{model_name} (H{horizon})\nTop {top_n} Feature Importances/Coefficients")
    ax.set_ylabel("Importance / Coefficient Value")
    plt.tight_layout()

# --- Helper for Target Variable Inference ---
def infer_target_variable_from_y_name(y_series_name: str):
    """
    Infers the base target variable name from a y_series name like 'base_target_target_h1'.
    Returns 'base_target' or None if parsing fails.
    """
    if not y_series_name or not isinstance(y_series_name, str):
        return None
    
    parts = y_series_name.split("_target_h")
    if len(parts) > 1: # Check if '_target_h' was found
        # Further check if the part after _target_h is a number (the horizon)
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
        print("\n--- Overall Performance Summary (from test3.py) ---")
        display(summary_df)

        # Compare to DummyRegressor
        dummy_rmses = summary_df[summary_df['Model'] == 'DummyRegressor'][['Horizon', 'RMSE']].set_index('Horizon')
        if not dummy_rmses.empty:
            print("\n--- RMSE Comparison to DummyRegressor ---")
            comparison_list = []
            for idx, row in summary_df.iterrows():
                if row['Model'] != 'DummyRegressor':
                    dummy_rmse = dummy_rmses.loc[row['Horizon'], 'RMSE'] if row['Horizon'] in dummy_rmses.index else np.nan
                    improvement = dummy_rmse - row['RMSE'] if not np.isnan(dummy_rmse) and not np.isnan(row['RMSE']) else np.nan
                    comparison_list.append({
                        'Horizon': row['Horizon'],
                        'Model': row['Model'],
                        'Model_RMSE': row['RMSE'],
                        'Dummy_RMSE': dummy_rmse,
                        'Improvement_vs_Dummy': improvement
                    })
            comparison_df = pd.DataFrame(comparison_list).sort_values(by=['Horizon', 'Model_RMSE'])
            display(comparison_df)


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
                        print(f"Successfully inferred target variable as: '{effective_target_variable}'")
                        break # Stop after first successful inference
    if not inferred_successfully:
        print(f"Could not infer target variable from y_test/y_train names. Using default: '{DEFAULT_TARGET_VARIABLE}'")

    for horizon in horizons_to_run:
        if horizon not in results_store:
            print(f"\n--- Horizon {horizon}: Not found in results_store. Skipping. ---")
            continue
        
        print(f"\n\n--- Processing Horizon: {horizon} ---")
        X_test, y_test, X_train, y_train = load_data_for_horizon(ARTIFACTS_BASE_DIR, horizon)

        if y_test is None or y_test.empty:
            print(f"  No y_test data for horizon {horizon}. Skipping detailed checks for this horizon.")
            continue
        
        # Extract dates from y_test index if it's a MultiIndex with 'date'
        test_dates = None
        if isinstance(y_test.index, pd.MultiIndex) and 'date' in y_test.index.names:
            test_dates = y_test.index.get_level_values('date')
        elif isinstance(y_test.index, pd.DatetimeIndex):
            test_dates = y_test.index
        else: # Fallback if dates are not easily extractable or not datetime
            test_dates = np.arange(len(y_test))


        models_in_horizon = results_store[horizon].keys()
        models_to_run_for_horizon = MODELS_TO_CHECK if MODELS_TO_CHECK else models_in_horizon

        for model_name in models_to_run_for_horizon:
            if model_name not in results_store[horizon]:
                print(f"  Model {model_name} not found for horizon {horizon}. Skipping.")
                continue
            if model_name == "Prophet": # Prophet needs special handling for predictions if X_test_orig is used
                print(f"  Skipping Prophet model ({model_name}) in this generic check script for now.")
                continue

            print(f"\n  --- Model: {model_name} (Horizon {horizon}) ---")
            model = load_model(ARTIFACTS_BASE_DIR, horizon, model_name)
            if model is None:
                print(f"    Could not load model {model_name}. Skipping checks.")
                continue

            # --- Test Data Checks ---
            if X_test is not None and not X_test.empty:
                predictions_test = get_predictions(model, X_test)
                if len(predictions_test) > 0:
                    _, residuals_test = summarize_predictions_and_errors(y_test, predictions_test, model_name, horizon, "Test")
                    if residuals_test is not None:
                         analyze_residuals_numerical(residuals_test, model_name, horizon, "Test")

                    # Visualizations for Test Data
                    num_plots = 5 # AvP_ts, AvP_scatter, ResVSPred, ResHist, ResQQ
                    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 5, 4))
                    fig.suptitle(f"Test Data Diagnostics: {model_name} - Horizon {horizon}", fontsize=16, y=1.03)
                    
                    plot_actual_vs_predicted(y_test, predictions_test, test_dates, model_name, horizon, axes[0], effective_target_variable, "Test")
                    plot_scatter_actual_vs_predicted(y_test, predictions_test, model_name, horizon, axes[1], effective_target_variable, "Test")
                    if residuals_test is not None:
                        plot_residuals_vs_predicted(residuals_test, predictions_test, model_name, horizon, axes[2], "Test")
                        plot_residuals_histogram(residuals_test, model_name, horizon, axes[3], "Test")
                        plot_residuals_qq(residuals_test, model_name, horizon, axes[4], "Test")
                    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
                    plt.show()

                    # Feature Importance Plot (if applicable)
                    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                        fig_fi, ax_fi = plt.subplots(1, 1, figsize=(10, 6))
                        plot_feature_importance(model, X_test.columns.tolist(), model_name, horizon, ax_fi)
                        plt.show()
                else:
                    print(f"    Could not generate test predictions for {model_name}.")
            else:
                print(f"    X_test_scaled not available for {model_name}, H{horizon}. Skipping test data checks.")

            # --- Train Data Checks (Optional, good for overfitting check) ---
            if X_train is not None and not X_train.empty and y_train is not None and not y_train.empty:
                print(f"\n    --- Train Data Checks for {model_name} (Horizon {horizon}) ---")
                predictions_train = get_predictions(model, X_train)
                if len(predictions_train) > 0:
                    _, residuals_train = summarize_predictions_and_errors(y_train, predictions_train, model_name, horizon, "Train")
                    if residuals_train is not None:
                        analyze_residuals_numerical(residuals_train, model_name, horizon, "Train")
                        # Optionally, add visual checks for train data too
                else:
                    print(f"    Could not generate train predictions for {model_name}.")
            else:
                print(f"    Train data (X_train_scaled or y_train) not available for H{horizon}. Skipping train data checks.")


if __name__ == "__main__":
    run_sanity_checks()
    print("\n--- Sanity Checks Completed ---")
