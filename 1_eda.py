import pandas as pd
from typing import List, Union, Tuple # Added for type hinting
import matplotlib.pyplot as plt
import numpy as np
import math

import regression_data_preparer
import importlib
importlib.reload(regression_data_preparer)  # Ensure the latest version of PanelDataPreparer is used
from regression_data_preparer import RegressionDataPreparer

TARGET_VARIABLES = {'interest_income_to_assets':'bank', 'interest_expense_to_assets':'bank',
                   'non_interest_income_to_assets':'bank', 'non_interest_expense_to_assets':'bank',
                   'net_charge_offs_to_loans_and_leases':'bank'}
FEATURE_VARIABLES = {'deposit_ratio':'bank', 'loan_to_asset_ratio':'bank', 'log_total_assets':'bank', 
                     'cpi_qoq':'macro',      'gdp_qoq':'macro',     'unemployment':'macro', 'household_delinq':'macro', 
                     'tbill_3m':'macro',     'tbill_10y':'macro',   'spread_10y_3m':'macro', 'sp500_qoq':'macro',
                     'corp_bond_spread':'macro', 'vix_qoq':'macro', 'is_structural_break':'bank',
                     # 'dep_small_3m_less_to_assets':'bank',
                     # 'dep_small_3m_1y_to_assets':'bank',
                     # 'dep_small_1y_3y_to_assets':'bank',
                     # 'dep_small_3y_more_to_assets':'bank',
                     # 'dep_large_3m_less_to_assets':'bank',
                     # 'dep_large_3m_1y_to_assets':'bank',
                     # 'dep_large_1y_3y_to_assets':'bank',
                     # 'dep_large_3y_more_to_assets':'bank', 
                     # 'closed_end_first_liens_1_4_res_prop_3m_less_to_assets':'bank',
                     # 'closed_end_first_liens_1_4_res_prop_3m_1y_to_assets':'bank',
                     # 'closed_end_first_liens_1_4_res_prop_1y_3y_to_assets':'bank',
                     # 'closed_end_first_liens_1_4_res_prop_3y_5y_to_assets':'bank',
                     # 'closed_end_first_liens_1_4_res_prop_5y_15y_to_assets':'bank',
                     # 'closed_end_first_liens_1_4_res_prop_15y_more_to_assets':'bank',
                     # 'all_other_loans_3m_less_to_assets':'bank',
                     # 'all_other_loans_3m_1y_to_assets':'bank',
                     # 'all_other_loans_1y_3y_to_assets':'bank',
                     # 'all_other_loans_3y_5y_to_assets':'bank',
                     # 'all_other_loans_5y_15y_to_assets':'bank',
                     # 'all_other_loans_15y_more_to_assets':'bank',
                     } # Added more macro vars
FORECAST_HORIZONS = list(range(1, 2))
# Get feature variables as a list


c = {
    'TARGET_VARIABLES': TARGET_VARIABLES, # Ensure this is a string             
    'FEATURE_VARIABLES': FEATURE_VARIABLES,             
    'INCLUDE_TIME_FE': True,                           
    'INCLUDE_BANK_FE': False,    
    'INCLUDE_STRUCTURAL_BREAK_DUMMY': True,               
    'OUTLIER_THRESHOLD_TARGET': 3.0,                    
    'MIN_OBS_PER_BANK': 12,             
    'CORRECT_STRUCTURAL_BREAKS_TOTAL_ASSETS': True,                
    'DATA_BEGIN': None, #'2017-01-01',                              
    'DATA_END': None,                                  
    'RESTRICT_TO_NUMBER_OF_BANKS': 500,                 
    'RESTRICT_TO_BANK_SIZE': None,                      
    'RESTRICT_TO_MINIMAL_DEPOSIT_RATIO': None,          
    'RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO': None,     
    'INCLUDE_AUTOREGRESSIVE_LAGS': True,                
    'NUMBER_OF_LAGS_TO_INCLUDE': 8,                     
    'TRAIN_TEST_SPLIT_DIMENSION': 'date',               
    'TEST_SPLIT': "2007-01-01"   # Takes a number or a date in the format 'YYYY-MM-DD'
}


print("--- Loading Data ---")
fred = pd.read_parquet('data/fred/macro_data_processed.parquet')
fdic = pd.read_parquet('data/fdic/fdic_data_processed.parquet').set_index(['id', 'date'])
yahoo = pd.read_parquet('data/yahoo/yahoo.parquet')
print("Data loaded.")


data_preparer = RegressionDataPreparer(fred, fdic, yahoo, config=c)


print("--- Missing value stats ----------------------------------------------------------------------------")
df0 = data_preparer.rectangularize_dataframe(data_preparer.df1).reset_index()

data_information = {}

# Number of banks in the full dataset
number_of_banks = df0['id'].nunique()
data_information['number_of_banks'] = number_of_banks
print(f"Number of banks in the dataset: {number_of_banks}")

# Number of positions in the full dataset
pd.read_parquet('data/fdic/fdic_data_extracted_raw_codes.parquet')

# Number of missing values in total assets
missing_total_assets = df0['log_total_assets'].isna().sum()
data_information['missing_log_total_assets'] = missing_total_assets
print(f"Number of missing values in log total assets: {missing_total_assets}")

# Number of banks with full data, i.e. no missing values in log total assets
banks_with_full_data = df0.groupby('id')['log_total_assets'].apply(lambda x: x.notna().all()).sum()
data_information['banks_with_full_data'] = banks_with_full_data
print(f"Number of banks with full data (no missing log total assets): {banks_with_full_data}")
# Number of banks with at least one missing value in log total assets
banks_with_missing_data = number_of_banks - banks_with_full_data
data_information['banks_with_missing_data'] = banks_with_missing_data
print(f"Number of banks with at least one missing value in log total assets: {banks_with_missing_data}")
data_information['percentage_banks_with_full_data'] = banks_with_full_data / number_of_banks * 100
print(f"Percentage of banks with full data: {data_information['percentage_banks_with_full_data']:.2f}%")

# Number of banks removed with not enough observations
nr_banks_removed_not_enough_data = data_preparer.data_processing_stats['bank_removal_not_enough_data']['banks_before'] - data_preparer.data_processing_stats['bank_removal_not_enough_data']['banks_after']
data_information['banks_removed_insufficient_obs'] = nr_banks_removed_not_enough_data
print(f"Number of banks removed due to not enough observations: {nr_banks_removed_not_enough_data}")

# Missing values in total assets over time
missing_total_assets_over_time = df0.groupby('date')['log_total_assets'].apply(lambda x: x.isna().sum())
# Simple plot of missing values in total assets over time
missing_total_assets_over_time.plot(title='Missing values in log total assets over time', ylabel='Number of missing values', xlabel='Date')
plt.show()
print(f"\nData Information Dictionary:\n{data_information}")




print("\n--- Plotting Target and Feature Variables Over Time ---------------------------------")

df_final = data_preparer.final_data

plot_variables = list(TARGET_VARIABLES.keys()) + list(FEATURE_VARIABLES.keys())

# Identify macro variables from the feature variables
bank_vars_set = list(var for var, var_type in FEATURE_VARIABLES.items() if var_type == 'bank') + list(TARGET_VARIABLES.keys())
macro_vars_set = list(var for var, var_type in FEATURE_VARIABLES.items() if var_type == 'macro')

num_plots = len(plot_variables)
num_cols = 3
num_rows = math.ceil(num_plots / num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4), sharex=True)
axes_flat = axes.flatten()

for i, var_name in enumerate(plot_variables):
    ax = axes_flat[i]
    if var_name not in df_final.columns:
        print(f"Warning: Variable '{var_name}' not found in df_final. Skipping plot.")
        ax.set_title(f"{var_name} (Not Found)")
        ax.text(0.5, 0.5, "Data N/A", ha="center", va="center", transform=ax.transAxes)
        continue

    if var_name in macro_vars_set:
        # Macro variable: plot directly (taking the first value per date)
        macro_data_ts = df_final.groupby('date')[var_name].mean()
        ax.plot(macro_data_ts.index, macro_data_ts.values, label=var_name, color='purple')
        ax.set_title(f"{var_name} (Macro)")
    else:
        # Bank-level variable: plot mean, median, and IQR
        grouped = df_final.groupby('date')[var_name]
        mean_ts = grouped.mean()
        median_ts = grouped.median()
        q1_ts = grouped.quantile(0.25)
        q3_ts = grouped.quantile(0.75)

        ax.plot(mean_ts.index, mean_ts.values, label='Mean', color='blue')
        ax.plot(median_ts.index, median_ts.values, label='Median', color='orange', linestyle='--')
        ax.fill_between(q1_ts.index, q1_ts.values, q3_ts.values, color='skyblue', alpha=0.4, label='IQR')
        ax.set_title(f"{var_name} (Bank-Level)")
    
    ax.legend(fontsize='small')
    ax.tick_params(axis='x', rotation=45)

for j in range(i + 1, len(axes_flat)): # Hide any unused subplots
    fig.delaxes(axes_flat[j])

fig.suptitle("Target and Feature Variables Over Time", fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
plt.show()


# Make a correlation plot of the target and feature variables
print("\n--- Correlation Plot of Target and Feature Variables ---------------------------------")
# Calculate the correlation matrix for the target and feature variables
correlation_vars = list(TARGET_VARIABLES.keys()) + list(FEATURE_VARIABLES.keys())
correlation_matrix = df_final[correlation_vars].corr()
# Plot the correlation matrix
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_vars)), correlation_vars, rotation=45, ha='right')
plt.yticks(range(len(correlation_vars)), correlation_vars)
plt.title("Correlation Matrix of Target and Feature Variables")
plt.tight_layout()
plt.show()
# Plot the correlation matrix as a heatmap
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap of Target and Feature Variables")
plt.tight_layout()
plt.show()

# --- Plot Maturity Structures ---
def plot_maturity_structure(df: pd.DataFrame):
    """
    Displays a stacked bar chart of deposit and loan maturities over time.
    """
    print("\n--- Plotting Deposit and Loan Maturity Structures ---------------------------------")
    
    # --- Data Preparation ---
    df_maturities = pd.DataFrame(index=df.index)

    # Sum deposit maturities
    df_maturities['Deposits < 3m'] = df['dep_small_3m_less_to_assets'] + df['dep_large_3m_less_to_assets']
    df_maturities['Deposits 3m-1y'] = df['dep_small_3m_1y_to_assets'] + df['dep_large_3m_1y_to_assets']
    df_maturities['Deposits 1y-3y'] = df['dep_small_1y_3y_to_assets'] + df['dep_large_1y_3y_to_assets']
    df_maturities['Deposits > 3y'] = df['dep_small_3y_more_to_assets'] + df['dep_large_3y_more_to_assets']
    
    # Sum loan maturities
    df_maturities['Loans < 3m'] = df['closed_end_first_liens_1_4_res_prop_3m_less_to_assets'] + df['all_other_loans_3m_less_to_assets']
    df_maturities['Loans 3m-1y'] = df['closed_end_first_liens_1_4_res_prop_3m_1y_to_assets'] + df['all_other_loans_3m_1y_to_assets']
    df_maturities['Loans 1y-3y'] = df['closed_end_first_liens_1_4_res_prop_1y_3y_to_assets'] + df['all_other_loans_1y_3y_to_assets']
    df_maturities['Loans 3y-5y'] = df['closed_end_first_liens_1_4_res_prop_3y_5y_to_assets'] + df['all_other_loans_3y_5y_to_assets']
    df_maturities['Loans 5y-15y'] = df['closed_end_first_liens_1_4_res_prop_5y_15y_to_assets'] + df['all_other_loans_5y_15y_to_assets']
    df_maturities['Loans > 15y'] = df['closed_end_first_liens_1_4_res_prop_15y_more_to_assets'] + df['all_other_loans_15y_more_to_assets']

    # Add total ratios for comparison
    df_maturities['Total Deposit Ratio'] = df['deposit_ratio']
    df_maturities['Total Loan Ratio'] = df['loan_to_asset_ratio']

    # Aggregate by date (mean across all banks)
    agg_maturities = df_maturities.groupby('date').mean()

    # --- Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Deposit Maturity Plot
    deposit_cols = ['Deposits < 3m', 'Deposits 3m-1y', 'Deposits 1y-3y', 'Deposits > 3y']
    agg_maturities[deposit_cols].plot(kind='bar', stacked=True, ax=axes[0], colormap='viridis')
    agg_maturities['Total Deposit Ratio'].plot(ax=axes[0], color='red', linestyle='--', marker='o', markersize=4, label='Total Deposit Ratio (for comparison)')
    axes[0].set_title('Deposit Maturity Structure (as % of Assets)')
    axes[0].set_ylabel('Ratio to Assets')
    axes[0].legend(title='Maturity Buckets')
    
    # Loan Maturity Plot
    loan_cols = ['Loans < 3m', 'Loans 3m-1y', 'Loans 1y-3y', 'Loans 3y-5y', 'Loans 5y-15y', 'Loans > 15y']
    agg_maturities[loan_cols].plot(kind='bar', stacked=True, ax=axes[1], colormap='plasma')
    agg_maturities['Total Loan Ratio'].plot(ax=axes[1], color='blue', linestyle='--', marker='o', markersize=4, label='Total Loan Ratio (for comparison)')
    axes[1].set_title('Loan Maturity Structure (as % of Assets)')
    axes[1].set_ylabel('Ratio to Assets')
    axes[1].legend(title='Maturity Buckets')

    # Format x-axis ticks to be more readable
    tick_labels = [item.strftime('%Y-Q%q') for item in agg_maturities.index]
    axes[1].set_xticklabels(tick_labels, rotation=45, ha='right')
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

if 'dep_small_3m_less_to_assets' in FEATURE_VARIABLES:
    plot_maturity_structure(df_final)

# Look at the training and test sets
print("\n--- Training and Test Sets Overview --------------------------------------------------")

X_train_scaled_df, X_test_scaled_df, y_train, y_test, X_train_orig, X_test_orig = data_preparer.get_horizon_specific_data(horizon=1,
                                                                                                                          target_variable='interest_income_to_assets')