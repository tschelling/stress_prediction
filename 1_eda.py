import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import importlib
from typing import List, Dict, Tuple, Optional, Union

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

# --- Helper function to prepare y for a specific target and horizon (from 5_multi_output_regressions.py) ---
def prepare_y_for_target_eda(base_data_df: pd.DataFrame, target_col_name: str, horizon: int,
                             X_idx: pd.MultiIndex):
    df_h = base_data_df.copy()
    shifted_target_col = f'{target_col_name}_target_h{horizon}'

    if target_col_name not in df_h.columns:
        print(f"Warning: Target column '{target_col_name}' not found in base data for y preparation.")
        empty_idx_name = X_idx.names if X_idx is not None else ['id', 'date']
        return pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=empty_idx_name))

    df_h[shifted_target_col] = df_h.groupby(level='id', group_keys=False)[target_col_name].shift(-horizon)
    df_h.dropna(subset=[shifted_target_col], inplace=True)

    if df_h.empty:
        print(f"Warning: DataFrame empty after shifting and NaN drop for target '{target_col_name}', H{horizon}.")
        empty_idx_name = X_idx.names if X_idx is not None else ['id', 'date']
        return pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=empty_idx_name))

    y_full_specific = df_h[shifted_target_col]
    y_aligned = pd.Series(dtype='float64')
    if X_idx is not None and not X_idx.empty:
        y_aligned = y_full_specific.reindex(X_idx).dropna()
    else:
        y_aligned = pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=y_full_specific.index.names))
    return y_aligned

# --- Plotting Helper Functions ---

def plot_distribution_combined(train_series: Optional[pd.Series], test_series: Optional[pd.Series], title: str, ax: plt.Axes):
    """Plots combined histogram for train and test series."""
    if ax is None:
        print(f"Skipping distribution plot for {title}: Axes not provided.")
        return

    if (train_series is None or train_series.empty) and (test_series is None or test_series.empty):
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} (Histogram)")
        return

    data_to_plot = []
    if train_series is not None and not train_series.empty:
        data_to_plot.append({'series': train_series.dropna(), 'label': 'Train', 'color': 'skyblue'})
    if test_series is not None and not test_series.empty:
        data_to_plot.append({'series': test_series.dropna(), 'label': 'Test', 'color': 'lightcoral'})

    if not data_to_plot:
        ax.text(0.5, 0.5, "No valid data to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} (Histogram)")
        return

    for item in data_to_plot:
        sns.histplot(item['series'], ax=ax, color=item['color'], label=item['label'], kde=True, stat="density", common_norm=False)
    ax.set_title(f"{title} (Histogram)")
    ax.legend()

def plot_timeseries_aggregated_combined(fig: plt.Figure, ax: plt.Axes,
                                        train_series: Optional[pd.Series], test_series: Optional[pd.Series],
                                        title: str, is_macro: bool = False):
    """Plots aggregated (median and IQR) or direct time series for train and test data."""
    if ax is None:
        print(f"Skipping timeseries plot for {title}: Axes not provided.")
        return

    plot_sets = [
        (train_series, 'Train', 'blue', '-'),
        (test_series, 'Test', 'red', '--'),
    ]
    any_data_plotted = False

    for series_data, label_prefix, color, style in plot_sets:
        if series_data is None or series_data.empty:
            print(f"Data for '{label_prefix}' in '{title}' is empty. Skipping.")
            continue

        if is_macro: # For macro, plot directly as it's unique per date
            if isinstance(series_data.index, pd.MultiIndex) and 'date' in series_data.index.names:
                # If macro is passed with multi-index, take first bank's series
                first_id = series_data.index.get_level_values('id')[0]
                data_to_plot = series_data.xs(first_id, level='id')
            elif isinstance(series_data.index, pd.DatetimeIndex):
                 data_to_plot = series_data
            else:
                print(f"Macro series '{title}' ({label_prefix}) has unexpected index. Skipping.")
                continue
            ax.plot(data_to_plot.index, data_to_plot, label=f'{label_prefix}', color=color, linestyle=style)
            any_data_plotted = True
        else: # For bank-specific, aggregate
            if not isinstance(series_data.index, pd.MultiIndex) or 'date' not in series_data.index.names:
                print(f"Skipping {label_prefix} for {title}: index is not MultiIndex with 'date'.")
                continue
            try:
                numeric_series = pd.to_numeric(series_data, errors='coerce').dropna()
                if numeric_series.empty:
                    print(f"Warning: Series for {label_prefix} ({title}) is empty after converting to numeric. Skipping.")
                    continue
                agg_data = numeric_series.groupby(level='date').agg(
                    q25=lambda x: x.quantile(0.25),
                    q50=lambda x: x.quantile(0.50),
                    q75=lambda x: x.quantile(0.75)
                )
                if agg_data.empty:
                    print(f"Warning: Empty aggregated data for {label_prefix} ({title}). Skipping.")
                    continue
                ax.plot(agg_data.index, agg_data['q50'], label=f'{label_prefix} Median', color=color, linestyle=style)
                ax.fill_between(agg_data.index, agg_data['q25'], agg_data['q75'], color=color, alpha=0.2)
                any_data_plotted = True
            except Exception as e:
                print(f"Error during aggregation/plotting for {label_prefix} ({title}): {e}")

    if not any_data_plotted:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.7)
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        if fig: fig.autofmt_xdate(rotation=45)
    except Exception as e_date:
        print(f"Warning: Could not format x-axis as dates for {title}: {e_date}")

def plot_timeseries_asset_weighted_vs_unweighted_combined(fig: plt.Figure, ax: plt.Axes,
                                                          train_df_eda: Optional[pd.DataFrame],
                                                          test_df_eda: Optional[pd.DataFrame],
                                                          value_col: str, asset_col: str, title: str):
    """Plots asset-weighted mean vs. unweighted median for a value column, for train and test."""
    if ax is None:
        print(f"Skipping weighted/unweighted plot for {title}: Axes not provided.")
        return

    plot_sets = [
        (train_df_eda, 'Train', 'blue', 'deepskyblue'),
        (test_df_eda, 'Test', 'red', 'lightcoral'),
    ]
    any_data_plotted = False

    for df_data, label_prefix, color_median, color_weighted in plot_sets:
        if df_data is None or df_data.empty or value_col not in df_data.columns or asset_col not in df_data.columns:
            print(f"Data for '{label_prefix}' in '{title}' (weighted/unweighted) is insufficient. Skipping.")
            continue

        # Unweighted Median
        if isinstance(df_data.index, pd.MultiIndex) and 'date' in df_data.index.names:
            try:
                median_ts = df_data[value_col].groupby(level='date').median()
                if not median_ts.empty:
                    ax.plot(median_ts.index, median_ts, label=f'{label_prefix} Median', color=color_median, linestyle='--')
                    any_data_plotted = True
            except Exception as e:
                print(f"Error plotting median for {label_prefix} ({title}): {e}")

            # Asset-Weighted Mean
            try:
                df_temp = df_data.copy()
                df_temp['weighted_value'] = df_temp[value_col] * df_temp[asset_col]
                sum_weighted_value = df_temp.groupby(level='date')['weighted_value'].sum()
                sum_assets = df_temp.groupby(level='date')[asset_col].sum()
                asset_weighted_mean_ts = sum_weighted_value / sum_assets
                if not asset_weighted_mean_ts.empty:
                    ax.plot(asset_weighted_mean_ts.index, asset_weighted_mean_ts, label=f'{label_prefix} Asset-Weighted Mean', color=color_weighted, linestyle='-')
                    any_data_plotted = True
            except Exception as e:
                print(f"Error plotting asset-weighted mean for {label_prefix} ({title}): {e}")
        else:
            print(f"Index for {label_prefix} ({title}) not suitable for weighted/unweighted aggregation.")

    if not any_data_plotted:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.legend(fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.7)
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        if fig: fig.autofmt_xdate(rotation=45)
    except Exception as e_date:
        print(f"Warning: Could not format x-axis as dates for {title}: {e_date}")


def plot_nan_heatmap_bucketed(df: pd.DataFrame, asset_col: str, title: str, ax: plt.Axes, num_buckets: int = 12):
    if df is None or df.empty or asset_col not in df.columns or ax is None:
        if ax: ax.text(0.5, 0.5, "No data or asset column for NaN heatmap", ha='center', va='center', transform=ax.transAxes)
        print(f"Skipping NaN heatmap for {title}: Data or Axes not provided.")
        return

    df_copy = df.copy()
    # Ensure 'date' is in datetime format for resampling
    if not isinstance(df_copy.index, pd.MultiIndex) or 'date' not in df_copy.index.names:
        print("NaN heatmap requires MultiIndex with 'date'. Skipping.")
        if ax: ax.text(0.5, 0.5, "Index not suitable", ha='center', va='center', transform=ax.transAxes)
        return

    # Create logarithmic asset buckets
    df_copy[asset_col] = pd.to_numeric(df_copy[asset_col], errors='coerce')
    df_copy.dropna(subset=[asset_col], inplace=True)
    if df_copy.empty:
        if ax: ax.text(0.5, 0.5, "No valid asset data for bucketing", ha='center', va='center', transform=ax.transAxes)
        return

    # Use log scale for buckets, handle non-positive assets
    min_asset = df_copy[asset_col][df_copy[asset_col] > 0].min()
    if pd.isna(min_asset): min_asset = 1e-9 # Fallback if all assets are <=0
    
    df_copy['log_assets'] = np.log(df_copy[asset_col].clip(lower=min_asset)) # Clip at min_asset to avoid log(0) or log(negative)
    
    # Create quantiles based on log_assets for better distribution in buckets
    try:
        df_copy['asset_bucket'] = pd.qcut(df_copy['log_assets'], q=num_buckets, labels=False, duplicates='drop')
    except ValueError as e: # Happens if too few unique log_asset values for qcut
        print(f"Warning: Could not create {num_buckets} asset buckets due to insufficient unique asset values: {e}. Trying fewer buckets or direct binning.")
        df_copy['asset_bucket'] = pd.cut(df_copy['log_assets'], bins=min(num_buckets, df_copy['log_assets'].nunique()), labels=False, duplicates='drop')

    # Calculate NaN percentage per asset bucket and resampled date (e.g., QuarterEnd)
    nan_summary_list = []
    features_to_check = [col for col in df_copy.columns if col not in ['log_assets', 'asset_bucket', asset_col] and col not in df_copy.index.names]

    for feature in features_to_check:
        # Group by date (resampled to QuarterEnd) and asset_bucket, then calculate NaN percentage
        # Reset index to make 'date' a column for resampling
        temp_df_reset = df_copy.reset_index()
        if 'date' not in temp_df_reset.columns:
            print("Error: 'date' column not found after reset_index. Skipping NaN heatmap.")
            if ax: ax.text(0.5, 0.5, "Date column error", ha='center', va='center', transform=ax.transAxes)
            return

        temp_df_reset['date'] = pd.to_datetime(temp_df_reset['date'])
        nan_pct = temp_df_reset.groupby([pd.Grouper(key='date', freq='Q'), 'asset_bucket'])[feature].apply(lambda x: x.isnull().mean() * 100)
        nan_pct_df = nan_pct.unstack(level='asset_bucket') # Pivot asset_bucket to columns
        nan_pct_df.columns = [f'Bucket {int(c) if pd.notna(c) else "NA"}' for c in nan_pct_df.columns] # Rename bucket columns
        nan_pct_df['feature'] = feature
        nan_summary_list.append(nan_pct_df.reset_index().melt(id_vars=['date', 'feature'], var_name='asset_bucket_label', value_name='nan_percentage'))

    if not nan_summary_list:
        if ax: ax.text(0.5, 0.5, "No NaN summary data", ha='center', va='center', transform=ax.transAxes)
        return

    final_nan_summary = pd.concat(nan_summary_list)
    heatmap_pivot = final_nan_summary.pivot_table(index='feature', columns=['date', 'asset_bucket_label'], values='nan_percentage')
    
    if heatmap_pivot.empty:
        if ax: ax.text(0.5, 0.5, "Pivot for heatmap is empty", ha='center', va='center', transform=ax.transAxes)
        return

    sns.heatmap(heatmap_pivot, ax=ax, cmap="YlGnBu", cbar_kws={'label': '% NaN'})
    ax.set_title(title)
    ax.set_xlabel("Date & Asset Bucket (Log Scale)")
    ax.set_ylabel("Feature")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=8)

def plot_correlation_heatmap(df: pd.DataFrame, title: str, ax: plt.Axes):
    if df is None or df.empty or ax is None:
        if ax: ax.text(0.5, 0.5, "No data for Corr heatmap", ha='center', va='center', transform=ax.transAxes)
        print(f"Skipping Correlation heatmap for {title}: Data or Axes not provided.")
        return
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        if ax: ax.text(0.5, 0.5, "Not enough numeric columns for Corr heatmap", ha='center', va='center', transform=ax.transAxes)
        print(f"Skipping Correlation heatmap for {title}: Not enough numeric columns.")
        return

    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, annot_kws={"size": 8})
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)


def summarize_panel_structure(df: pd.DataFrame, title_prefix: str, id_col='id', date_col='date'):
    print(f"\n--- Panel Data Structure Summary: {title_prefix} ---")
    if df is None or df.empty:
        print("DataFrame is empty. No structure to summarize.")
        return

    num_unique_banks = df[id_col].nunique()
    min_date, max_date = df[date_col].min(), df[date_col].max()
    total_obs = len(df)
    print(f"Number of unique banks: {num_unique_banks}")
    print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"Total observations: {total_obs}")

    obs_per_bank = df.groupby(id_col).size()
    print("\nObservations per bank (summary):")
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

    # Check for balanced panel
    if num_unique_banks > 0:
        is_balanced = all(obs_per_bank == obs_per_bank.iloc[0]) and \
                      active_banks_over_time.nunique() == 1 and \
                      active_banks_over_time.iloc[0] == num_unique_banks
        print(f"\nIs the panel balanced? {'Yes' if is_balanced else 'No'}")
    else:
        print("\nPanel balance cannot be determined (no banks).")


# --- Main EDA Function ---
def perform_eda():
    print("--- Starting Exploratory Data Analysis ---")

    # 1. Load and Prepare Data
    print("\n--- 1. Loading and Preparing Data ---")
    try:
        raw_data_full = pd.read_parquet('data.parquet')
        print(f"Raw data loaded. Shape: {raw_data_full.shape}")
    except Exception as e:
        print(f"Error loading data.parquet: {e}. EDA cannot proceed.")
        return

    data_preparer = RegressionDataPreparer(initial_df=raw_data_full, config=config_dict_preparer_eda)
    if data_preparer.base_data_for_horizons is None or data_preparer.base_data_for_horizons.empty:
        print("Base data preparation by RegressionDataPreparer failed or resulted in empty data. EDA cannot proceed.")
        return
    
    # Get unscaled features from preparer (it uses PRIMARY_TARGET_FOR_PREPARER for its internal y)
    # The y_train/y_test from get_horizon_specific_data are for PRIMARY_TARGET_FOR_PREPARER,
    # we will prepare our specific EDA targets manually.
    prepared_data_for_X = data_preparer.get_horizon_specific_data(horizon=EDA_FORECAST_HORIZON)
    if prepared_data_for_X is None:
        print(f"Failed to prepare X data for EDA (Horizon {EDA_FORECAST_HORIZON}). EDA cannot proceed.")
        return
    _, _, _, _, X_train_unscaled_df, X_test_unscaled_df = prepared_data_for_X

    if X_train_unscaled_df is None or X_train_unscaled_df.empty:
        print("X_train_unscaled_df is empty. EDA cannot proceed with train data.")
        # Optionally, could still proceed with test data if available and desired
        return

    # Create base EDA DataFrames with features
    train_df_eda = X_train_unscaled_df.copy()
    test_df_eda = X_test_unscaled_df.copy() if X_test_unscaled_df is not None else pd.DataFrame(index=train_df_eda.index[:0], columns=train_df_eda.columns) # Empty df with same columns if test is None

    # Manually prepare and add each EDA target variable
    print(f"\nPreparing target variables for EDA (Horizon {EDA_FORECAST_HORIZON})...")
    base_data_for_targets = data_preparer.base_data_for_horizons.copy() # Use the cleaned base for target shifting

    for target_name in TARGETS_FOR_EDA:
        y_train_target = prepare_y_for_target_eda(base_data_for_targets, target_name, EDA_FORECAST_HORIZON, train_df_eda.index)
        train_df_eda[target_name] = y_train_target # Adds as a new column

        if not test_df_eda.empty:
            y_test_target = prepare_y_for_target_eda(base_data_for_targets, target_name, EDA_FORECAST_HORIZON, test_df_eda.index)
            test_df_eda[target_name] = y_test_target
        else: # Ensure column exists in empty test_df_eda
            test_df_eda[target_name] = pd.Series(dtype='float64')
            
        # Drop NaNs introduced by this specific target shifting, only for this target column
        # This keeps rows that might be valid for other targets or features
        train_df_eda.dropna(subset=[target_name], inplace=True)
        if not test_df_eda.empty:
             test_df_eda.dropna(subset=[target_name], inplace=True)


    # Merge Asset Column (if exists and configured)
    if ASSET_COLUMN_NAME and ASSET_COLUMN_NAME in base_data_for_targets.columns:
        print(f"Merging asset column '{ASSET_COLUMN_NAME}' for weighted averages...")
        train_df_eda = train_df_eda.merge(base_data_for_targets[[ASSET_COLUMN_NAME]], left_index=True, right_index=True, how='left')
        if not test_df_eda.empty:
            test_df_eda = test_df_eda.merge(base_data_for_targets[[ASSET_COLUMN_NAME]], left_index=True, right_index=True, how='left')
    else:
        print(f"Asset column '{ASSET_COLUMN_NAME}' not found or not configured. Skipping asset-weighted analysis.")


    # 2. Overall Data Overview
    print("\n--- 2. Overall Data Overview (Train Set) ---")
    print(f"Train EDA DataFrame shape: {train_df_eda.shape}")
    if not test_df_eda.empty:
        print(f"Test EDA DataFrame shape: {test_df_eda.shape}")
    else:
        print("Test EDA DataFrame is empty.")

    print("\nTrain EDA DataFrame Head:")
    print(train_df_eda.head())
    print("\nTrain EDA DataFrame Info:")
    train_df_eda.info()
    print("\nTrain EDA DataFrame Descriptive Statistics:")
    print(train_df_eda.describe().T)

    # 3. Panel Data Structure Analysis
    if not train_df_eda.empty:
        summarize_panel_structure(train_df_eda.reset_index(), "Train Set")
        fig_nan_train, ax_nan_train = plt.subplots(1, 1, figsize=(15, 10)) # Adjusted size
        plot_nan_heatmap_bucketed(train_df_eda, ASSET_COLUMN_NAME, "Missing Values Heatmap (Train Set - Bucketed by Assets)", ax_nan_train)
        plt.show()

    if not test_df_eda.empty:
        summarize_panel_structure(test_df_eda.reset_index(), "Test Set")
        fig_nan_test, ax_nan_test = plt.subplots(1, 1, figsize=(15, 10)) # Adjusted size
        plot_nan_heatmap_bucketed(test_df_eda, ASSET_COLUMN_NAME, "Missing Values Heatmap (Test Set - Bucketed by Assets)", ax_nan_test)
        plt.show()


    # 4. Target Variables Analysis
    print("\n--- 4. Target Variables Analysis ---")
    num_targets = len(TARGETS_FOR_EDA)
    fig_targets_dist, axes_targets_dist = plt.subplots(num_targets, 1, figsize=(8, 4 * num_targets), squeeze=False) # Changed to 1 column for hist only
    fig_targets_ts, axes_targets_ts = plt.subplots(num_targets, 1, figsize=(14, 5 * num_targets), squeeze=False)
    
    if ASSET_COLUMN_NAME and ASSET_COLUMN_NAME in train_df_eda.columns :
        fig_targets_weighted_ts, axes_targets_weighted_ts = plt.subplots(num_targets, 1, figsize=(14, 5 * num_targets), squeeze=False)
    else:
        fig_targets_weighted_ts, axes_targets_weighted_ts = None, None


    for i, target_name in enumerate(TARGETS_FOR_EDA):
        print(f"\nAnalyzing Target: {target_name}")
        plot_distribution_combined(train_df_eda.get(target_name), test_df_eda.get(target_name),
                                   target_name, axes_targets_dist[i, 0]) # Only one ax for hist
        
        plot_timeseries_aggregated_combined(fig_targets_ts, axes_targets_ts[i, 0],
                                            train_df_eda.get(target_name), test_df_eda.get(target_name),
                                            f"{target_name} Over Time (Median & IQR)")
        
        if fig_targets_weighted_ts and target_name in KEY_FINANCIAL_RATIOS_FOR_WEIGHTED_AVG:
            plot_timeseries_asset_weighted_vs_unweighted_combined(
                fig_targets_weighted_ts, axes_targets_weighted_ts[i,0],
                train_df_eda, test_df_eda,
                target_name, ASSET_COLUMN_NAME,
                f"{target_name}: Asset-Weighted Mean vs. Median"
            )
        elif fig_targets_weighted_ts: # Hide unused subplot
            axes_targets_weighted_ts[i,0].set_visible(False)


    fig_targets_dist.suptitle("Target Variable Distributions (Train vs. Test)", fontsize=16, y=1.01)
    fig_targets_dist.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    fig_targets_ts.suptitle("Target Variables Over Time (Train vs. Test)", fontsize=16, y=1.01)
    fig_targets_ts.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    if fig_targets_weighted_ts:
        fig_targets_weighted_ts.suptitle("Target Variables: Weighted vs. Unweighted (Train vs. Test)", fontsize=16, y=1.01)
        fig_targets_weighted_ts.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    # 5. Feature Analysis
    print("\n--- 5. Feature Analysis ---")
    bank_specific_features = [f for f in FEATURE_VARIABLES if f not in MACRO_FEATURES_LIST]
    
    # Small Multiples Time Series
    print("\nFeature Time Series (Small Multiples)...")
    num_bank_features = len(bank_specific_features)
    fig_bank_feat_ts, axes_bank_feat_ts = plt.subplots(num_bank_features, 1, figsize=(14, 4 * num_bank_features), squeeze=False)
    for i, feat_name in enumerate(bank_specific_features):
        plot_timeseries_aggregated_combined(fig_bank_feat_ts, axes_bank_feat_ts[i, 0],
                                            train_df_eda.get(feat_name), test_df_eda.get(feat_name),
                                            f"Bank Feature: {feat_name} (Median & IQR)")
    fig_bank_feat_ts.suptitle("Bank-Specific Features Over Time", fontsize=16, y=1.01)
    fig_bank_feat_ts.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()

    num_macro_features = len(MACRO_FEATURES_LIST)
    fig_macro_feat_ts, axes_macro_feat_ts = plt.subplots(num_macro_features, 1, figsize=(14, 4 * num_macro_features), squeeze=False)
    for i, feat_name in enumerate(MACRO_FEATURES_LIST):
        plot_timeseries_aggregated_combined(fig_macro_feat_ts, axes_macro_feat_ts[i, 0],
                                            train_df_eda.get(feat_name), test_df_eda.get(feat_name),
                                            f"Macro Feature: {feat_name}", is_macro=True)
    fig_macro_feat_ts.suptitle("Macroeconomic Features Over Time", fontsize=16, y=1.01)
    fig_macro_feat_ts.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()

    # Individual Feature Distributions
    print("\nFeature Distributions...")
    num_all_features = len(FEATURE_VARIABLES)
    num_cols_feat_dist = 3
    num_rows_feat_dist = (num_all_features + num_cols_feat_dist - 1) // num_cols_feat_dist
    fig_feat_dist, axes_feat_dist_flat = plt.subplots(num_rows_feat_dist, num_cols_feat_dist, figsize=(5 * num_cols_feat_dist, 4 * num_rows_feat_dist), squeeze=False)
    axes_feat_dist_flat = axes_feat_dist_flat.flatten()

    for i, feat_name in enumerate(FEATURE_VARIABLES):
        if i < len(axes_feat_dist_flat): # Ensure we don't go out of bounds for axes
            plot_distribution_combined(train_df_eda.get(feat_name), test_df_eda.get(feat_name),
                                       feat_name, axes_feat_dist_flat[i]) # Only one ax for hist
    fig_feat_dist.suptitle("Feature Distributions (Train vs. Test)", fontsize=16, y=1.01)
    fig_feat_dist.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()

    # Asset-Weighted vs. Unweighted for Key Bank-Specific Features
    if ASSET_COLUMN_NAME and ASSET_COLUMN_NAME in train_df_eda.columns:
        print("\nAsset-Weighted vs. Unweighted Averages for Key Bank Features...")
        key_bank_features_for_weighted = [f for f in KEY_FINANCIAL_RATIOS_FOR_WEIGHTED_AVG if f in bank_specific_features]
        if key_bank_features_for_weighted:
            num_key_bank_weighted = len(key_bank_features_for_weighted)
            fig_bank_weighted_ts, axes_bank_weighted_ts = plt.subplots(num_key_bank_weighted, 1, figsize=(14, 5 * num_key_bank_weighted), squeeze=False)
            for i, feat_name in enumerate(key_bank_features_for_weighted):
                plot_timeseries_asset_weighted_vs_unweighted_combined(
                    fig_bank_weighted_ts, axes_bank_weighted_ts[i,0],
                    train_df_eda, test_df_eda,
                    feat_name, ASSET_COLUMN_NAME,
                    f"Bank Feature {feat_name}: Asset-Weighted Mean vs. Median"
                )
            fig_bank_weighted_ts.suptitle("Key Bank Features: Weighted vs. Unweighted", fontsize=16, y=1.01)
            fig_bank_weighted_ts.tight_layout(rect=[0, 0, 1, 0.99])
            plt.show()
        else:
            print("No key bank-specific features configured or found for weighted average plots.")


    # 6. Relationship Analysis
    print("\n--- 6. Relationship Analysis (Train Set) ---")
    # Correlation Heatmap (numeric features + all EDA targets)
    numeric_cols_for_corr = FEATURE_VARIABLES + TARGETS_FOR_EDA
    numeric_cols_for_corr = [col for col in numeric_cols_for_corr if col in train_df_eda.columns and pd.api.types.is_numeric_dtype(train_df_eda[col])]
    
    if numeric_cols_for_corr:
        fig_corr, ax_corr = plt.subplots(1, 1, figsize=(max(12, len(numeric_cols_for_corr)*0.5), max(10, len(numeric_cols_for_corr)*0.4)))
        plot_correlation_heatmap(train_df_eda[numeric_cols_for_corr], "Feature and Target Correlation Matrix (Train Set)", ax_corr)
        plt.show()

    # Scatter Plots (Key Features vs. Primary Target)
    print("\nScatter Plots (Key Features vs. Primary Target)...")
    primary_eda_target = COMPARISON_TARGET_NI_EDA # or PRIMARY_TARGET_FOR_PREPARER
    key_scatter_features = [f for f in bank_specific_features if f in train_df_eda.columns][:3] # Top 3 bank features

    if primary_eda_target in train_df_eda.columns and key_scatter_features:
        num_scatter = len(key_scatter_features)
        fig_scatter, axes_scatter = plt.subplots(1, num_scatter, figsize=(7 * num_scatter, 6), squeeze=False)
        fig_scatter.suptitle(f"Scatter Plots: Features vs. {primary_eda_target} (Train & Test)", fontsize=16, y=1.02)
        for i, feat_name in enumerate(key_scatter_features):
            ax_sc = axes_scatter[0,i]
            # Combine train and test for hue-based plotting
            temp_train_scatter = train_df_eda[[feat_name, primary_eda_target]].copy().dropna()
            temp_train_scatter['Dataset'] = 'Train'
            temp_test_scatter = pd.DataFrame()
            if not test_df_eda.empty and feat_name in test_df_eda.columns and primary_eda_target in test_df_eda.columns:
                temp_test_scatter = test_df_eda[[feat_name, primary_eda_target]].copy().dropna()
                temp_test_scatter['Dataset'] = 'Test'
            
            combined_scatter_df = pd.concat([temp_train_scatter, temp_test_scatter])

            if not combined_scatter_df.empty:
                sns.scatterplot(data=combined_scatter_df, x=feat_name, y=primary_eda_target, hue='Dataset', ax=ax_sc, palette={'Train': 'blue', 'Test': 'red'}, alpha=0.6, s=20)
                ax_sc.set_title(f"{feat_name} vs. {primary_eda_target}")
                ax_sc.grid(True, linestyle=':', alpha=0.7)
            else:
                ax_sc.text(0.5,0.5, "No data for scatter", ha='center', va='center', transform=ax_sc.transAxes)
                ax_sc.set_title(f"{feat_name} vs. {primary_eda_target}")

        fig_scatter.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    else:
        print(f"Primary EDA target '{primary_eda_target}' or key scatter features not available in train_df_eda.")

    print("\n--- EDA Script Finished ---")


if __name__ == "__main__":
    # Create a dummy data.parquet for testing if it doesn't exist
    # This dummy data creation is simplified and might not cover all edge cases
    # or perfectly match the structure of your actual data.parquet.
    if not os.path.exists('data.parquet'):
        print("Creating dummy data.parquet for testing EDA script...")
        rng = np.random.default_rng(42)
        n_banks = 10 # Smaller for faster dummy creation
        n_dates_per_bank = 20
        dates = pd.to_datetime(pd.date_range('2015-01-01', periods=n_dates_per_bank, freq='Q'))
        ids = [f'bank_dummy_{j}' for j in range(n_banks)]
        multi_index = pd.MultiIndex.from_product([ids, dates], names=['id', 'date'])
        
        dummy_data = pd.DataFrame(index=multi_index)
        
        # Add all features and targets defined in configurations
        all_vars_needed = list(set(FEATURE_VARIABLES + TARGETS_FOR_EDA + [ASSET_COLUMN_NAME, PRIMARY_TARGET_FOR_PREPARER]))
        
        for var_name in all_vars_needed:
            if var_name == ASSET_COLUMN_NAME: # e.g. 'total_assets'
                 dummy_data[var_name] = rng.lognormal(mean=np.log(1e9), sigma=1.5, size=len(multi_index))
            elif var_name == 'log_total_assets': # Will be derived if total_assets exists
                 if ASSET_COLUMN_NAME in dummy_data:
                     dummy_data[var_name] = np.log(dummy_data[ASSET_COLUMN_NAME].replace(0, 1e-9)) # Avoid log(0)
            elif 'ratio' in var_name or 'to_assets' in var_name or 'delinq' in var_name: # Ratios often between 0 and 1, or small
                 dummy_data[var_name] = rng.uniform(0.001, 0.9, size=len(multi_index))
            elif var_name in MACRO_FEATURES_LIST: # Macro vars
                 macro_values = rng.normal(0.01, 0.02, size=len(dates))
                 dummy_data[var_name] = dummy_data.index.get_level_values('date').map(pd.Series(macro_values, index=dates))
            else: # Other general numeric features
                 dummy_data[var_name] = rng.normal(10, 5, size=len(multi_index))
        
        # Ensure PRIMARY_TARGET_FOR_PREPARER exists for RegressionDataPreparer
        if PRIMARY_TARGET_FOR_PREPARER not in dummy_data.columns:
            dummy_data[PRIMARY_TARGET_FOR_PREPARER] = rng.normal(0.02, 0.005, size=len(multi_index))

        dummy_data.to_parquet('data.parquet')
        print("Dummy data.parquet created for EDA.")

    perform_eda()
