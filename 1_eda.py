import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import importlib
from typing import List, Dict, Tuple, Optional, Union

#region --- Imports and Configuration ---
import regression_data_preparer
importlib.reload(regression_data_preparer) # For development, ensure latest version
from regression_data_preparer import RegressionDataPreparer

# --- Configuration (Adapted from 5_multi_output_regressions.py) ---
PRIMARY_TARGET_FOR_PREPARER = 'interest_income_to_assets'
FEATURE_VARIABLES = ['gdp_qoq', 'deposit_ratio', 'loan_to_asset_ratio', 'log_total_assets', 'cpi_qoq', 'unemployment',
                     'household_delinq', 'tbill_3m', 'tbill_10y', 'spread_10y_3m', 'sp500_qoq',
                     'corp_bond_spread', 'vix_qoq']

MULTI_OUTPUT_COMPONENTS_EDA = { # Renamed to avoid conflict if 5_ is imported
    'interest_income_to_assets': {'sign': +1},
    'non_interest_income_to_assets': {'sign': +1},
    'interest_expense_to_assets': {'sign': -1},
    'non_interest_expense_to_assets': {'sign': -1}
}
COMPARISON_TARGET_NI_EDA = 'net_income_to_assets'

TARGETS_FOR_EDA = list(MULTI_OUTPUT_COMPONENTS_EDA.keys()) + [COMPARISON_TARGET_NI_EDA]

config_dict_preparer_eda = {
    'TARGET_VARIABLE': PRIMARY_TARGET_FOR_PREPARER, # Used by DataPreparer for AR lags if any
    'FEATURE_VARIABLES': FEATURE_VARIABLES,
    'INCLUDE_TIME_FE': False, # EDA usually on raw features before FE creation
    'INCLUDE_BANK_FE': False, # EDA usually on raw features before FE creation
    'OUTLIER_THRESHOLD_TARGET': 3.0,
    'MIN_OBS_PER_BANK': 10,
    'DATA_BEGIN': None,
    'DATA_END': None,
    'RESTRICT_TO_NUMBER_OF_BANKS': 2000,
    'RESTRICT_TO_BANK_SIZE': None,
    'RESTRICT_TO_MINIMAL_DEPOSIT_RATIO': None,
    'RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO': 0.1,
    'INCLUDE_AUTOREGRESSIVE_LAGS': False, # For EDA, we typically look at original features and targets
    'NUMBER_OF_LAGS_TO_INCLUDE': 0,    # No AR lags for EDA features unless specifically desired
    'TRAIN_TEST_SPLIT_DIMENSION': 'date',
    'TEST_SPLIT': 0.2
}

# --- EDA Specific Configuration ---
EDA_FORECAST_HORIZON = 1 # Analyze data as if preparing for 1-step ahead forecast
MACRO_FEATURES_LIST = ['gdp_qoq', 'cpi_qoq', 'unemployment', 'household_delinq',
                       'tbill_3m', 'tbill_10y', 'spread_10y_3m', 'sp500_qoq',
                       'corp_bond_spread', 'vix_qoq']
ASSET_COLUMN_NAME = 'total_assets' # Must exist in the raw data loaded by RegressionDataPreparer
KEY_FINANCIAL_RATIOS_FOR_WEIGHTED_AVG = [
    'deposit_ratio', 'loan_to_asset_ratio', # Features
    'interest_income_to_assets', 'net_income_to_assets' # Targets
]

#endregion

#region --- Helper Functions ---
# --- Helper function to prepare y for a specific target and horizon (from 5_multi_output_regressions.py) ---
def prepare_y_for_target_eda(base_data_df: pd.DataFrame, target_col_name: str, horizon: int,
                             X_idx: pd.MultiIndex):
    df_h = base_data_df.copy()
    shifted_target_col = f'{target_col_name}_target_h{horizon}' # Ensure this column name is unique if target_col_name itself could end with _h{number}
    df_h[shifted_target_col] = df_h.groupby(level='id', group_keys=False)[target_col_name].shift(-horizon)
    df_h.dropna(subset=[shifted_target_col], inplace=True)

    y_full_specific = df_h[shifted_target_col]
    y_aligned = y_full_specific.reindex(X_idx).dropna()
    return y_aligned


# --- Plotting Helper Functions ---

def plot_distribution_combined(train_series: Optional[pd.Series], test_series: Optional[pd.Series], title: str, ax: plt.Axes):
    """Plots combined histogram for train and test series."""
    data_to_plot = []
    if train_series is not None and not train_series.empty:
        data_to_plot.append({'series': train_series.dropna(), 'label': 'Train', 'color': 'skyblue'})
    if test_series is not None and not test_series.empty:
        data_to_plot.append({'series': test_series.dropna(), 'label': 'Test', 'color': 'lightcoral'})

    for item in data_to_plot:
        sns.histplot(item['series'], ax=ax, color=item['color'], label=item['label'], kde=True, stat="density", common_norm=False)
    ax.set_title(f"{title} (Histogram)")
    ax.legend()

def plot_timeseries_aggregated_combined(fig: plt.Figure, ax: plt.Axes,
                                        train_series: Optional[pd.Series], test_series: Optional[pd.Series],
                                        title: str, is_macro: bool = False):
    """Plots aggregated (median and IQR) or direct time series for train and test data."""
    plot_sets = [
        (train_series, 'Train', 'blue', '-'),
        (test_series, 'Test', 'red', '--'),
    ]

    for series_data, label_prefix, color, style in plot_sets:
        if series_data is None or series_data.empty:
            continue

        if is_macro:
            if isinstance(series_data.index, pd.MultiIndex):
                first_id = series_data.index.get_level_values('id')[0]
                data_to_plot = series_data.xs(first_id, level='id')
            else: # Assumed to be DatetimeIndex or will error
                 data_to_plot = series_data
            ax.plot(data_to_plot.index, data_to_plot, label=f'{label_prefix}', color=color, linestyle=style)
        else:
            numeric_series = pd.to_numeric(series_data, errors='coerce').dropna()
            agg_data = numeric_series.groupby(level='date').agg(
                q25=lambda x: x.quantile(0.25),
                q50=lambda x: x.quantile(0.50),
                q75=lambda x: x.quantile(0.75)
            )
            ax.plot(agg_data.index, agg_data['q50'], label=f'{label_prefix} Median', color=color, linestyle=style)
            ax.fill_between(agg_data.index, agg_data['q25'], agg_data['q75'], color=color, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if fig: fig.autofmt_xdate(rotation=45)


def plot_timeseries_asset_weighted_vs_unweighted_combined(fig: plt.Figure, ax: plt.Axes,
                                                          train_df_eda: Optional[pd.DataFrame],
                                                          test_df_eda: Optional[pd.DataFrame],
                                                          value_col: str, asset_col: str, title: str):
    """Plots asset-weighted mean vs. unweighted median for a value column, for train and test."""
    plot_sets = [
        (train_df_eda, 'Train', 'blue', 'deepskyblue'),
        (test_df_eda, 'Test', 'red', 'lightcoral'),
    ]

    for df_data, label_prefix, color_median, color_weighted in plot_sets:
        if df_data is None or df_data.empty or value_col not in df_data.columns or asset_col not in df_data.columns:
            continue

        # Unweighted Median
        median_ts = df_data[value_col].groupby(level='date').median()
        ax.plot(median_ts.index, median_ts, label=f'{label_prefix} Median', color=color_median, linestyle='--')

        # Asset-Weighted Mean
        df_temp = df_data.copy()
        df_temp['weighted_value'] = df_temp[value_col] * df_temp[asset_col]
        sum_weighted_value = df_temp.groupby(level='date')['weighted_value'].sum()
        sum_assets = df_temp.groupby(level='date')[asset_col].sum()
        asset_weighted_mean_ts = sum_weighted_value / sum_assets
        ax.plot(asset_weighted_mean_ts.index, asset_weighted_mean_ts, label=f'{label_prefix} Asset-Weighted Mean', color=color_weighted, linestyle='-')

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.legend(fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if fig: fig.autofmt_xdate(rotation=45)

def plot_correlation_heatmap(df: pd.DataFrame, title: str, ax: plt.Axes):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, annot_kws={"size": 8})
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)


def summarize_panel_structure(df: pd.DataFrame, title_prefix: str, id_col='id', date_col='date'):
    print(f"\n--- Panel Data Structure Summary: {title_prefix} ---")
    num_unique_banks = df[id_col].nunique()
    min_date, max_date = df[date_col].min(), df[date_col].max()
    total_obs = len(df)
    print(f"Number of unique banks: {num_unique_banks}")
    min_date_str = min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else "N/A"
    max_date_str = max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else "N/A"
    print(f"Date range: {min_date_str} to {max_date_str}")
    print(f"Total observations: {total_obs}")

    obs_per_bank = df.groupby(id_col).size()
    print("\nObservations per bank (summary):")
    if obs_per_bank.empty:
        print("  No data to describe (e.g., panel is empty or contains no banks).")
    else:
        print(obs_per_bank.describe())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Panel Structure: {title_prefix}", fontsize=14, y=1.02)
    sns.histplot(obs_per_bank, ax=axes[0], kde=False)
    axes[0].set_title("Distribution of Observations per Bank")
    axes[0].set_xlabel("Number of Observations")
    axes[0].set_ylabel("Number of Banks")

    active_banks_over_time = df.groupby(date_col)[id_col].nunique()
    axes[1].plot(active_banks_over_time.index, active_banks_over_time.values)
    axes[1].set_title("Number of Active Banks Over Time")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Number of Unique Banks")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
#endregion







#region Execute




print("--- Starting Exploratory Data Analysis ---")
#region --- 1. Load and Prepare Data ---
# 1. Load and Prepare Data
print("\n--- 1. Loading and Preparing Data ---")
raw_data_full = pd.read_parquet('data.parquet')
print(f"Raw data loaded. Shape: {raw_data_full.shape}")

data_preparer = RegressionDataPreparer(initial_df=raw_data_full, config=config_dict_preparer_eda)
# Get unscaled features from preparer (it uses PRIMARY_TARGET_FOR_PREPARER for its internal y)
# The y_train/y_test from get_horizon_specific_data are for PRIMARY_TARGET_FOR_PREPARER,
# we will prepare our specific EDA targets manually.
prepared_data_for_X = data_preparer.get_horizon_specific_data(horizon=EDA_FORECAST_HORIZON)
_, _, _, _, X_train_unscaled_df, X_test_unscaled_df = prepared_data_for_X

# If X_train_unscaled_df is None or empty, subsequent operations will fail,
# which is acceptable per the request to remove explicit checks.


# Create base EDA DataFrames with features
train_df_eda = X_train_unscaled_df.copy()
test_df_eda = X_test_unscaled_df.copy() if X_test_unscaled_df is not None else pd.DataFrame(index=train_df_eda.index[:0], columns=train_df_eda.columns) # Empty df with same columns if test is None

# Manually prepare and add each EDA target variable
print(f"\nPreparing target variables for EDA (Horizon {EDA_FORECAST_HORIZON})...")
base_data_for_targets = data_preparer.base_data_for_horizons.copy() # Use the cleaned base for target shifting

for target_name in TARGETS_FOR_EDA:
    y_train_target = prepare_y_for_target_eda(base_data_for_targets, target_name, EDA_FORECAST_HORIZON, train_df_eda.index)
    train_df_eda[target_name] = y_train_target # Adds as a new column

    y_test_target = prepare_y_for_target_eda(base_data_for_targets, target_name, EDA_FORECAST_HORIZON, test_df_eda.index)
    test_df_eda[target_name] = y_test_target
        
    # Drop NaNs introduced by this specific target shifting, only for this target column
    # This keeps rows that might be valid for other targets or features
    train_df_eda.dropna(subset=[target_name], inplace=True)
    test_df_eda.dropna(subset=[target_name], inplace=True)

# Merge Asset Column (if exists and configured)
train_df_eda = train_df_eda.merge(base_data_for_targets[[ASSET_COLUMN_NAME]], left_index=True, right_index=True, how='left')
test_df_eda = test_df_eda.merge(base_data_for_targets[[ASSET_COLUMN_NAME]], left_index=True, right_index=True, how='left')

#endregion
#region --- 2. Overall Data Overview ---
# 2. Overall Data Overview
print("\n--- 2. Overall Data Overview (Train Set) ---")
print(f"Train EDA DataFrame shape: {train_df_eda.shape}")
print(f"Test EDA DataFrame shape: {test_df_eda.shape}")
print("\nTrain EDA DataFrame Head:")
print(train_df_eda.head())
print("\nTrain EDA DataFrame Info:")
train_df_eda.info()
print("\nTrain EDA DataFrame Descriptive Statistics:")
print(train_df_eda.describe().T)

#endregion
#region --- 3. Panel Data Structure Analysis ---
# 3. Panel Data Structure Analysis
summarize_panel_structure(train_df_eda.reset_index(), "Train Set")
summarize_panel_structure(test_df_eda.reset_index(), "Test Set")

#endregion
#region --- 4. Target Variables Analysis ---
# 4. Target Variables Analysis
print("\n--- 4. Target Variables Analysis ---")
num_targets = len(TARGETS_FOR_EDA)
num_cols_targets = 3
num_rows_targets = (num_targets + num_cols_targets - 1) // num_cols_targets

fig_targets_dist, axes_targets_dist_flat = plt.subplots(num_rows_targets, num_cols_targets, figsize=(5 * num_cols_targets, 4 * num_rows_targets), squeeze=False)
axes_targets_dist_flat = axes_targets_dist_flat.flatten()
fig_targets_ts, axes_targets_ts_flat = plt.subplots(num_rows_targets, num_cols_targets, figsize=(6 * num_cols_targets, 5 * num_rows_targets), squeeze=False, sharex=True)
axes_targets_ts_flat = axes_targets_ts_flat.flatten()
fig_targets_weighted_ts, axes_targets_weighted_ts_flat = plt.subplots(num_rows_targets, num_cols_targets, figsize=(6 * num_cols_targets, 5 * num_rows_targets), squeeze=False, sharex=True)
axes_targets_weighted_ts_flat = axes_targets_weighted_ts_flat.flatten()

for i, target_name in enumerate(TARGETS_FOR_EDA):
    print(f"\nAnalyzing Target: {target_name}")
    # Plot Distribution
    plot_distribution_combined(train_df_eda.get(target_name), test_df_eda.get(target_name),
                                target_name, axes_targets_dist_flat[i])
    
    plot_timeseries_aggregated_combined(fig_targets_ts, axes_targets_ts_flat[i],
                                        train_df_eda.get(target_name), test_df_eda.get(target_name),
                                        f"{target_name} Over Time (Median & IQR)")
    
    plot_timeseries_asset_weighted_vs_unweighted_combined(
        fig_targets_weighted_ts, axes_targets_weighted_ts_flat[i],
        train_df_eda, test_df_eda,
        target_name, ASSET_COLUMN_NAME,
        f"{target_name}: Asset-Weighted Mean vs. Median"
    )

for j in range(num_targets, len(axes_targets_dist_flat)): # Hide unused distribution subplots
    fig_targets_dist.delaxes(axes_targets_dist_flat[j])
for j in range(num_targets, len(axes_targets_ts_flat)): # Hide unused time series subplots
    fig_targets_ts.delaxes(axes_targets_ts_flat[j])
for j in range(num_targets, len(axes_targets_weighted_ts_flat)): # Hide unused weighted time series subplots
    fig_targets_weighted_ts.delaxes(axes_targets_weighted_ts_flat[j])

fig_targets_dist.suptitle("Target Variable Distributions (Train vs. Test)", fontsize=16, y=1.01)
fig_targets_dist.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
fig_targets_ts.suptitle("Target Variables Over Time (Train vs. Test)", fontsize=16, y=1.01)
fig_targets_ts.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
#region --- 5. Feature Analysis ---
# 5. Feature Analysis
print("\n--- 5. Feature Analysis ---")
bank_specific_features = [f for f in FEATURE_VARIABLES if f not in MACRO_FEATURES_LIST]

# Combined Small Multiples Time Series for ALL features
print("\nAll Feature Time Series (Small Multiples)...")
num_all_features_ts = len(FEATURE_VARIABLES)
num_cols_all_feat_ts = 3
num_rows_all_feat_ts = (num_all_features_ts + num_cols_all_feat_ts - 1) // num_cols_all_feat_ts

fig_all_feat_ts, axes_all_feat_ts_flat = plt.subplots(num_rows_all_feat_ts, num_cols_all_feat_ts, 
                                                        figsize=(6 * num_cols_all_feat_ts, 5 * num_rows_all_feat_ts), 
                                                        squeeze=False, sharex=True)
axes_all_feat_ts_flat = axes_all_feat_ts_flat.flatten()

for i, feat_name in enumerate(FEATURE_VARIABLES):
    is_macro_feature = feat_name in MACRO_FEATURES_LIST
    plot_timeseries_aggregated_combined(fig_all_feat_ts, axes_all_feat_ts_flat[i],
                                        train_df_eda.get(feat_name), test_df_eda.get(feat_name),
                                        f"Feature: {feat_name}", is_macro=is_macro_feature)
fig_all_feat_ts.suptitle("All Features Over Time (Median & IQR or Direct for Macro)", fontsize=16, y=1.01)
fig_all_feat_ts.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()

# Individual Feature Distributions
print("\nFeature Distributions...")
num_all_features = len(FEATURE_VARIABLES)
num_cols_feat_dist = 3
num_rows_feat_dist = (num_all_features + num_cols_feat_dist - 1) // num_cols_feat_dist
fig_feat_dist, axes_feat_dist_flat = plt.subplots(num_rows_feat_dist, num_cols_feat_dist, figsize=(5 * num_cols_feat_dist, 4 * num_rows_feat_dist), squeeze=False)
axes_feat_dist_flat = axes_feat_dist_flat.flatten()

for i, feat_name in enumerate(FEATURE_VARIABLES):
    plot_distribution_combined(train_df_eda.get(feat_name), test_df_eda.get(feat_name),
                                feat_name, axes_feat_dist_flat[i])
fig_feat_dist.suptitle("Feature Distributions (Train vs. Test)", fontsize=16, y=1.01)
fig_feat_dist.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()

# Asset-Weighted vs. Unweighted for Key Bank-Specific Features
print("\nAsset-Weighted vs. Unweighted Averages for Key Bank Features...")
key_bank_features_for_weighted = [f for f in KEY_FINANCIAL_RATIOS_FOR_WEIGHTED_AVG if f in bank_specific_features]

num_key_bank_weighted = len(key_bank_features_for_weighted)
num_cols_key_bank_weighted = 3
num_rows_key_bank_weighted = (num_key_bank_weighted + num_cols_key_bank_weighted - 1) // num_cols_key_bank_weighted

fig_bank_weighted_ts, axes_bank_weighted_ts_flat = plt.subplots(
    num_rows_key_bank_weighted, num_cols_key_bank_weighted,
    figsize=(6 * num_cols_key_bank_weighted, 5 * num_rows_key_bank_weighted),
    squeeze=False, sharex=True
)
axes_bank_weighted_ts_flat = axes_bank_weighted_ts_flat.flatten()

for i, feat_name in enumerate(key_bank_features_for_weighted):
    plot_timeseries_asset_weighted_vs_unweighted_combined(
        fig_bank_weighted_ts, axes_bank_weighted_ts_flat[i],
        train_df_eda, test_df_eda,
        feat_name, ASSET_COLUMN_NAME,
        f"Bank Feature {feat_name}: Asset-Weighted Mean vs. Median"
    )
for j in range(num_key_bank_weighted, len(axes_bank_weighted_ts_flat)): # Hide unused subplots
    fig_bank_weighted_ts.delaxes(axes_bank_weighted_ts_flat[j])
    
fig_bank_weighted_ts.suptitle("Key Bank Features: Weighted vs. Unweighted", fontsize=16, y=1.01)
fig_bank_weighted_ts.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()

#endregion
#region --- 6. Relationship Analysis ---
# 6. Relationship Analysis
print("\n--- 6. Relationship Analysis (Train Set) ---")
# Correlation Heatmap (numeric features + all EDA targets)
numeric_cols_for_corr = FEATURE_VARIABLES + TARGETS_FOR_EDA
numeric_cols_for_corr = [col for col in numeric_cols_for_corr if col in train_df_eda.columns and pd.api.types.is_numeric_dtype(train_df_eda[col])]
fig_corr, ax_corr = plt.subplots(1, 1, figsize=(max(12, len(numeric_cols_for_corr)*0.5), max(10, len(numeric_cols_for_corr)*0.4)))
plot_correlation_heatmap(train_df_eda[numeric_cols_for_corr], "Feature and Target Correlation Matrix (Train Set)", ax_corr)
plt.show()

# Scatter Plots (Key Features vs. Primary Target)
print("\nScatter Plots (Key Features vs. Primary Target)...")
primary_eda_target = COMPARISON_TARGET_NI_EDA # or PRIMARY_TARGET_FOR_PREPARER
key_scatter_features = [f for f in bank_specific_features if f in train_df_eda.columns][:3] # Top 3 bank features

num_scatter = len(key_scatter_features)
fig_scatter, axes_scatter = plt.subplots(1, num_scatter, figsize=(7 * num_scatter, 6), squeeze=False)
fig_scatter.suptitle(f"Scatter Plots: Features vs. {primary_eda_target} (Train & Test)", fontsize=16, y=1.02)
for i, feat_name in enumerate(key_scatter_features):
    ax_sc = axes_scatter[0,i]
    # Combine train and test for hue-based plotting
    temp_train_scatter = train_df_eda[[feat_name, primary_eda_target]].copy().dropna()
    temp_train_scatter['Dataset'] = 'Train'
    
    temp_test_scatter = test_df_eda[[feat_name, primary_eda_target]].copy().dropna()
    temp_test_scatter['Dataset'] = 'Test'
    
    combined_scatter_df = pd.concat([temp_train_scatter, temp_test_scatter])

    sns.scatterplot(data=combined_scatter_df, x=feat_name, y=primary_eda_target, hue='Dataset', ax=ax_sc, palette={'Train': 'blue', 'Test': 'red'}, alpha=0.6, s=20)
    ax_sc.set_title(f"{feat_name} vs. {primary_eda_target}")
    ax_sc.grid(True, linestyle=':', alpha=0.7)

fig_scatter.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n--- EDA Script Finished ---")
#endregion

#endregion

