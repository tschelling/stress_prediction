import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from publication_name_mapping import PUBLICATION_NAMES

def _add_latex_line_breaks_for_headers(latex_string: str) -> str:
    """Replaces ' / ' in headers with a LaTeX line break using makecell."""
    for name in PUBLICATION_NAMES.values():
        if ' / ' in name:
            new_name = name.replace(' / ', r' \\ / ')
            latex_string = latex_string.replace(name, f'\\makecell{{{new_name}}}')
    return latex_string

# --- Configuration ---
ARTIFACTS_DIRS = ["models_and_results_standard", "models_and_results_rfe"]

def plot_feature_importance(model, feature_names, ax=None, top_n=10, r2_score_val=None, model_display_name=None):
    """
    Plots feature importances or coefficients for a given model on a specific axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    title_prefix = model_display_name if model_display_name else "Feature"
    title_suffix = f" (R²: {r2_score_val:.2f})" if r2_score_val is not None and not np.isnan(r2_score_val) else ""

    importances_values = None
    processed_feature_names = feature_names

    if isinstance(model, Pipeline) and 'rfe' in model.named_steps:
        rfe_step = model.named_steps['rfe']
        final_estimator_name = model.steps[-1][0]
        regressor_step = model.named_steps[final_estimator_name]

        if hasattr(rfe_step, 'support_'):
            processed_feature_names = np.array(feature_names)[rfe_step.support_]
            if hasattr(regressor_step, 'feature_importances_'):
                importances_values = regressor_step.feature_importances_
            elif hasattr(regressor_step, 'coef_'):
                importances_values = regressor_step.coef_
            ax.set_title(f'{title_prefix} Importances (RFE){title_suffix}', fontsize=9)
        else:
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
        importances_values = importances_values.flatten()
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
        ax.set_title(f'{title_prefix} Importance N/A{title_suffix}', fontsize=9)

def get_model_information():
    """
    Scans specified artifact directories, loads models and test data,
    and calculates the RMSE for each model.
    """
    all_results = []

    print("--- Starting RMSE Calculation ---")

    for base_dir in ARTIFACTS_DIRS:
        if not os.path.exists(base_dir):
            print(f"Warning: Directory not found, skipping: {base_dir}")
            continue

        run_type = 'RFE' if 'rfe' in base_dir.lower() else 'Standard'
        print(f"\nProcessing directory: {base_dir} (Run Type: {run_type})")

        for target_dir_name in sorted(os.listdir(base_dir)):
            if not target_dir_name.startswith("target_"):
                continue
            
            target_variable_raw = target_dir_name.replace("target_", "")
            target_variable = PUBLICATION_NAMES.get(target_variable_raw, target_variable_raw)
            target_path = os.path.join(base_dir, target_dir_name)

            if not os.path.isdir(target_path):
                continue

            print(f"  Target: {target_variable}")

            for horizon_dir_name in sorted(os.listdir(target_path)):
                if not horizon_dir_name.startswith("horizon_"):
                    continue
                
                horizon = int(horizon_dir_name.replace("horizon_", ""))
                horizon_path = os.path.join(target_path, horizon_dir_name)
                if not os.path.isdir(horizon_path):
                    continue

                # --- Load Test Data for this Horizon ---
                data_dir = os.path.join(horizon_path, "data")
                X_test_path = os.path.join(data_dir, "X_test_scaled.parquet")
                y_test_path = os.path.join(data_dir, "y_test.parquet")

                if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
                    print(f"    Horizon {horizon}: Skipping, missing X_test or y_test data.")
                    continue

                X_test = pd.read_parquet(X_test_path)
                X_train = pd.read_parquet(os.path.join(data_dir, "X_train_scaled.parquet"))
                y_test_df = pd.read_parquet(y_test_path)
                y_test = y_test_df[y_test_df.columns[0]]
                y_train = pd.read_parquet(os.path.join(data_dir, "y_train.parquet"))

                # --- Iterate through models in the horizon directory ---
                for model_filename in sorted(os.listdir(horizon_path)):
                    if not model_filename.endswith(".joblib") or model_filename == "results_store.joblib":
                        continue

                    model_name = model_filename.replace(".joblib", "")
                    model_path = os.path.join(horizon_path, model_filename)
                    data_dir_path = os.path.join(horizon_path, "data")

                    # Load model
                    model = joblib.load(model_path)
                    
                    # Make predictions
                    predictions = model.predict(X_test)
                    
                    # Calculate Scores
                    mse  = mean_squared_error(y_test, predictions)
                    rmse = np.sqrt(mse)
                    mae  = np.mean(np.abs(y_test - predictions))

                     # Handle potential division by zero for MAPE
                    if (y_test == 0).any():
                        mape = np.mean(np.abs((y_test - predictions) / y_test.replace(0, np.nan))) * 100
                    else:
                        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                                            
                    r2   = model.score(X_test, y_test)
                    
                    # Nr of NAs in training and test data
                    num_nas_train = X_test.isna().sum().sum()
                    num_nas_test  = y_test.isna().sum()

                    # Nr banks and dates
                    num_banks_test = X_test.index.get_level_values('id').nunique()
                    num_banks_train = X_train.index.get_level_values('id').nunique()
                    avg_dates_per_bank_test = X_test.reset_index().groupby('id')['date'].nunique().mean()
                    avg_dates_per_bank_train = X_train.reset_index().groupby('id')['date'].nunique().mean()

                    # Hyperparameters and model information
                    hyperparameters = model.get_params()

                    # Number of features the model uses
                    if hasattr(model, 'named_steps') and 'rfe' in model.named_steps:
                        # It's a pipeline with an RFE step.
                        # Get the number of selected features by summing the support mask.
                        rfe_step = model.named_steps['rfe']
                        if hasattr(rfe_step, 'support_'):
                            num_features = rfe_step.support_.sum()
                        else:
                            # Fallback if RFE step is not fitted
                            num_features = np.nan
                    elif hasattr(model, 'n_features_in_'):
                        # It's a standard estimator, get the number of input features
                        num_features = model.n_features_in_
                    else:
                        num_features = X_test.shape[1]
                    # Store result
                    all_results.append({
                        'RunType': run_type,
                        'Target': target_variable,
                        'Horizon': horizon,
                        'Model': model_name,
                        'ModelPath': model_path,
                        'DataDir': data_dir_path,
                        'Hyperparameters': hyperparameters,
                        'RMSE': rmse,
                        'MSE': mse,
                        'MAE': mae,
                        'MAPE': mape,
                        'R2': r2,
                        'NumFeatures': num_features,
                        'NumNAsTrain': num_nas_train,
                        'NumNAsTest': num_nas_test,
                        'NumBanksTest': num_banks_test,
                        'NumBanksTrain': num_banks_train,
                        'StartDateTrain': X_train.index.get_level_values('date').min(),
                        'EndDateTrain': X_train.index.get_level_values('date').max(),
                        'StartDateTest': X_test.index.get_level_values('date').min(),
                        'EndDateTest': X_test.index.get_level_values('date').max(),
                        'AvgDatesPerBankTest': avg_dates_per_bank_test,
                        'AvgDatesPerBankTrain': avg_dates_per_bank_train
                    })

                    

    all_results = pd.DataFrame(all_results)
    all_results.sort_values(by=['RunType', 'Target', 'Horizon', 'RMSE'], inplace=True)
    return all_results


model_information = get_model_information()

# --- Create and Display Pivot Tables ---
print("\n--- Generating Comparison Tables (for Horizon 1) ---")

# Filter for a single horizon to make the pivot tables 2D
# This assumes horizon 1 is the primary one of interest for this comparison
horizon_1_data = model_information[model_information['Horizon'] == 1].copy()

# Add a 'BaseModel' column for grouping RFE and Standard models
horizon_1_data['BaseModel'] = horizon_1_data['Model'].str.replace('_RFE', '', regex=False)

# --- Number of Features Table ---
print("\n\n--- Table: Number of Features Used by Model ---")
features_pivot = horizon_1_data.pivot_table(
    index=['BaseModel', 'RunType'],
    columns='Target',
    values='NumFeatures'
)
features_pivot = features_pivot.fillna('-') # For clarity
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
    print(features_pivot)

# Save to LaTeX
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
features_pivot_latex = features_pivot.to_latex(
    caption='Number of Features Used by Model (Horizon 1)',
    label='tab:num_features',
    longtable=True,
    escape=True,
    float_format="%.0f"
)
features_pivot_latex = _add_latex_line_breaks_for_headers(features_pivot_latex)
features_table_path = os.path.join(output_dir, "num_features_table.tex")
with open(features_table_path, "w") as f:
    f.write(features_pivot_latex)
print(f"Saved number of features table to {features_table_path}")


# --- Hyperparameters Table ---
print("\n\n--- Table: Hyperparameters by Model ---")

import re

def _format_hyperparams_str(s: str) -> str:
    """Formats np.float64 values in a string representation of a dict."""
    def repl(match):
        try:
            val = float(match.group(1))
            return f"{val:.4g}"  # Use 4 significant figures for readability
        except (ValueError, IndexError):
            return match.group(0)  # Return original if conversion fails
    return re.sub(r"np\.float64\((.*?)\)", repl, s)

# Convert the dictionary of hyperparameters to a string for display
horizon_1_data['Hyperparameters_str'] = horizon_1_data['Hyperparameters'].astype(str).apply(_format_hyperparams_str)

hyperparams_pivot = horizon_1_data.pivot_table(
    index='Target',
    columns=['BaseModel', 'RunType'],
    values='Hyperparameters_str',
    aggfunc='first'  # Use 'first' as each combination should be unique for a given horizon
)
hyperparams_pivot = hyperparams_pivot.fillna('N/A')

# Display the full table without truncation
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
    print(hyperparams_pivot)

# Save to LaTeX
hyperparams_pivot_latex = hyperparams_pivot.to_latex(
    caption='Hyperparameters by Model (Horizon 1)',
    label='tab:hyperparams',
    longtable=True,
    escape=True
)
hyperparams_pivot_latex = _add_latex_line_breaks_for_headers(hyperparams_pivot_latex)
hyperparams_table_path = os.path.join(output_dir, "hyperparameters_table.tex")
with open(hyperparams_table_path, "w") as f:
    f.write(hyperparams_pivot_latex)
print(f"Saved hyperparameters table to {hyperparams_table_path}")


# --- Create Faceted Bar Chart ---
print("\n--- Generating Performance Chart ---")

# Define the desired order for the target variables
target_order = [
    'Interest Income / Assets',
    'Interest Expense / Assets',
    'Non-Interest Income / Assets',
    'Non-Interest Expense / Assets',
    'Net Charge-Offs / Loans'
]

# Convert 'Target' column to a categorical type with the specified order
# This ensures the facets in the plot appear in the desired order.
model_information['Target'] = pd.Categorical(
    model_information['Target'],
    categories=target_order,
    ordered=True
)
model_information.dropna(subset=['Target'], inplace=True)
model_information.sort_values(by=['Target', 'RunType', 'RMSE'], inplace=True)

# Add a 'BaseModel' column for grouping RFE and Standard models
model_information['BaseModel'] = model_information['Model'].str.replace('_RFE', '', regex=False)

# Create a FacetGrid. With 5 targets, col_wrap=3 results in a 2-row layout.
g = sns.FacetGrid(
    model_information,
    col='Target',
    col_wrap=2,
    height=5,
    aspect=1.2,
    sharex=False,
    sharey=False
)

# Define a function to plot on each facet, ordering models by their average RMSE
def plot_rmse_bars(data, **kwargs):
    # Sort models on the y-axis by the minimum RMSE between their Standard and RFE runs
    model_order = data.groupby('BaseModel')['RMSE'].min().sort_values(ascending=True).index
    ax = plt.gca()
    sns.barplot(x='RMSE', y='BaseModel', hue='RunType', data=data, order=model_order, orient='h', ax=ax)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Annotate bars with MAPE and R2.
    # This manual annotation approach is more robust than using bar_label
    # when dealing with grouped bars and potential NaNs.
    
    # Get the y-tick positions and their corresponding labels (model names)
    y_ticks_positions = {label.get_text(): pos for pos, label in zip(ax.get_yticks(), ax.get_yticklabels())}

    # Determine the order of hues (RunType) used by seaborn
    hue_order = [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else data['RunType'].unique()
    num_hues = len(hue_order)
    bar_height_total = 0.8  # Default width of a bar group in matplotlib
    bar_height_single = bar_height_total / num_hues

    for model_name in model_order:
        y_base = y_ticks_positions.get(model_name)
        if y_base is None: continue

        for i, run_type in enumerate(hue_order):
            row = data[(data['BaseModel'] == model_name) & (data['RunType'] == run_type)]
            if not row.empty and pd.notna(row['RMSE'].iloc[0]):
                rmse_val = row['RMSE'].iloc[0]
                r2_val = row['R2'].iloc[0]
                y_pos = y_base - (bar_height_total / 2) + (bar_height_single / 2) + (i * bar_height_single)
                if pd.notna(r2_val):
                    ax.text(rmse_val / 2, y_pos, f"R²: {r2_val:.2f}", ha='center', va='center', color='white', fontsize=10)

g.map_dataframe(plot_rmse_bars)
g.set_titles("Target: {col_name}")
g.set_xlabels("RMSE")
g.set_ylabels("Model")
g.add_legend(title='Run Type')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
# Save chart
chart_filename = "model_performance_comparison.png"
chart_path = os.path.join("plots", chart_filename)
os.makedirs(os.path.dirname(chart_path), exist_ok=True)
g.savefig(chart_path, bbox_inches='tight')

# --- Generate Feature Importance Charts ---
print("\n--- Generating Feature Importance Charts ---")

MODELS_FOR_FI_PLOT = ["XGBoost", "RandomForest", "DecisionTree", "Ridge"]
TOP_N_FEATURES = 15
HORIZON_TO_PLOT = 1

fi_data = model_information[model_information['Horizon'] == HORIZON_TO_PLOT]
unique_targets = sorted(fi_data['Target'].unique())

for target_name in unique_targets:
    target_df = fi_data[fi_data['Target'] == target_name]
    
    available_models = sorted([m for m in MODELS_FOR_FI_PLOT if m in target_df['BaseModel'].unique()])
    if not available_models:
        continue

    fig, axes = plt.subplots(
        nrows=len(available_models), 
        ncols=2, 
        figsize=(14, 5 * len(available_models)), 
        squeeze=False
    )

    for i, base_model_name in enumerate(available_models):
        # --- Standard Model (Left) ---
        ax_std = axes[i, 0]
        std_model_info = target_df[(target_df['BaseModel'] == base_model_name) & (target_df['RunType'] == 'Standard')]
        
        if not std_model_info.empty:
            info = std_model_info.iloc[0]
            model = joblib.load(info['ModelPath'])
            X_train = pd.read_parquet(os.path.join(info['DataDir'], "X_train_scaled.parquet"))
            feature_names = X_train.columns.tolist()
            plot_feature_importance(model, feature_names, ax=ax_std, top_n=TOP_N_FEATURES, r2_score_val=info['R2'], model_display_name=f"{base_model_name} (Standard)")
        else:
            ax_std.text(0.5, 0.5, "Standard model not available", ha='center', va='center')
            ax_std.set_title(f"{base_model_name} (Standard)")

        # --- RFE Model (Right) ---
        ax_rfe = axes[i, 1]
        rfe_model_info = target_df[(target_df['BaseModel'] == base_model_name) & (target_df['RunType'] == 'RFE')]

        if not rfe_model_info.empty:
            info = rfe_model_info.iloc[0]
            model = joblib.load(info['ModelPath'])
            X_train = pd.read_parquet(os.path.join(info['DataDir'], "X_train_scaled.parquet"))
            feature_names = X_train.columns.tolist()
            plot_feature_importance(model, feature_names, ax=ax_rfe, top_n=TOP_N_FEATURES, r2_score_val=info['R2'], model_display_name=f"{base_model_name} (RFE)")
        else:
            ax_rfe.text(0.5, 0.5, "RFE model not available", ha='center', va='center')
            ax_rfe.set_title(f"{base_model_name} (RFE)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    chart_filename = f"feature_importance_{target_name.replace(' / ', '_').replace(' ', '_').lower()}_h{HORIZON_TO_PLOT}.png"
    chart_path = os.path.join("plots", chart_filename)
    fig.savefig(chart_path, bbox_inches='tight')
    print(f"  Saved feature importance chart to {chart_path}")
    plt.close(fig)
