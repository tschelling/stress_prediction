import pandas as pd
from typing import List, Union, Tuple # Added for type hinting
import matplotlib.pyplot as plt
import numpy as np
import math
import os

import regression_data_preparer
import importlib
importlib.reload(regression_data_preparer)  # Ensure the latest version of PanelDataPreparer is used
from regression_data_preparer import RegressionDataPreparer
from publication_name_mapping import PUBLICATION_NAMES

TARGET_VARIABLES = {'interest_income_to_assets':'bank', 'interest_expense_to_assets':'bank',
                   'non_interest_income_to_assets':'bank', 'non_interest_expense_to_assets':'bank',
                   'net_charge_offs_to_loans_and_leases':'bank'}
FEATURE_VARIABLES = {'deposit_ratio':'bank', 'loan_to_asset_ratio':'bank', 'dep_demand_to_assets':'bank',
                     'log_total_assets':'bank', 
                     'cpi_qoq':'macro',      'gdp_qoq':'macro',     'unemployment_diff':'macro', 'household_delinq_diff':'macro', 
                     'tbill_3m_diff':'macro',     'tbill_10y_diff':'macro', 'sp500_qoq':'macro',
                     'corp_bond_spread_diff':'macro', 'vix_qoq':'macro', 'is_structural_break':'bank',
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
    'RESTRICT_TO_NUMBER_OF_BANKS': None, 
    'RESTRICT_TO_LARGEST_BANKS': 200, # Restrict to the largest N banks by average total assets                
    'RESTRICT_TO_BANK_SIZE': None,                      
    'RESTRICT_TO_MINIMAL_DEPOSIT_RATIO': None,          
    'RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO': None,     
    'INCLUDE_AUTOREGRESSIVE_LAGS': True,                
    'NUMBER_OF_LAGS_TO_INCLUDE': 4,                     
    'TRAIN_TEST_SPLIT_DIMENSION': 'date',               
    'TEST_SPLIT': 0.2   # Takes a number or a date in the format 'YYYY-MM-DD'
}


print("--- Loading Data ---")
fred = pd.read_parquet('data/fred/macro_data_processed.parquet')
fdic = pd.read_parquet('data/fdic/fdic_data_processed.parquet').set_index(['id', 'date'])
yahoo = pd.read_parquet('data/yahoo/yahoo.parquet')
print("Data loaded.")


reg_data = RegressionDataPreparer(fred, fdic, yahoo, config=c)

#region Missing value stats
print("--- Missing value stats ----------------------------------------------------------------------------")
df0 = reg_data.rectangularize_dataframe(reg_data.df1).reset_index()

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
nr_banks_removed_not_enough_data = reg_data.data_processing_stats['bank_removal_not_enough_data']['banks_before'] - reg_data.data_processing_stats['bank_removal_not_enough_data']['banks_after']
data_information['banks_removed_insufficient_obs'] = nr_banks_removed_not_enough_data
print(f"Number of banks removed due to not enough observations: {nr_banks_removed_not_enough_data}")

# Missing values in total assets over time
missing_total_assets_over_time = df0.groupby('date')['log_total_assets'].apply(lambda x: x.isna().sum())
# Simple plot of missing values in total assets over time
missing_total_assets_over_time.plot(title='Missing values in log total assets over time', ylabel='Number of missing values', xlabel='Date')
plt.show()
print(f"\nData Information Dictionary:\n{data_information}")


#endregion

#region Descriptive statistics per variable

show_variables = list(TARGET_VARIABLES.keys()) + list(FEATURE_VARIABLES.keys())

print("\n--- Descriptive Statistics per Variable -------------------------------------------------------------")
temp = reg_data.final_data[show_variables].stack().rename_axis(index=['id', 'date', 'variable']).rename('value')
temp = pd.DataFrame(temp.astype(float))
descriptive_statistics = temp.dropna().groupby('variable').describe()

print("\n--- Latex ---")
# Separate descriptive statistics for bank and macro variables
bank_vars = [var for var, var_type in TARGET_VARIABLES.items() if var_type == 'bank'] + \
            [var for var, var_type in FEATURE_VARIABLES.items() if var_type == 'bank']
macro_vars = [var for var, var_type in FEATURE_VARIABLES.items() if var_type == 'macro']

# Ensure all variables in descriptive_statistics are categorized
descriptive_statistics_bank = descriptive_statistics.loc[descriptive_statistics.index.intersection(bank_vars)]
descriptive_statistics_macro = descriptive_statistics.loc[descriptive_statistics.index.intersection(macro_vars)]

# Create 'tex' directory if it doesn't exist
tex_dir = 'tex'
os.makedirs(tex_dir, exist_ok=True)

# Function to save DataFrame to LaTeX
def save_df_to_latex(df, filename, caption, label):
    df_renamed = df.rename(index=PUBLICATION_NAMES)
    filepath = os.path.join(tex_dir, filename)
    df_renamed.to_latex(filepath,
                caption=caption, 
                label=label, 
                float_format="%.2f", # Format numbers to two decimal places
                column_format="l" + "c" * len(df.columns), # Left-align index, center-align columns
                longtable=True, # Use longtable for long tables
                index=True # Include index (variable names)
               )
    print(f"Saved LaTeX table to {filepath}")

# Save bank variables table
save_df_to_latex(descriptive_statistics_bank, 
                 'descriptive_stats_bank_vars.tex', 
                 'Descriptive Statistics for Bank-Specific Variables', 
                 'tab:desc_stats_bank')

# Save macro variables table
save_df_to_latex(descriptive_statistics_macro, 
                 'descriptive_stats_macro_vars.tex', 
                 'Descriptive Statistics for Macroeconomic Variables', 
                 'tab:desc_stats_macro')


#endregion



#region Plot variables over time
print("\n--- Plotting Target and Feature Variables Over Time ---------------------------------")

df_final = reg_data.final_data

# Identify macro variables from the feature variables
bank_vars_set = list(TARGET_VARIABLES.keys()) + list(var for var, var_type in FEATURE_VARIABLES.items() if var_type == 'bank') 
macro_vars_set = list(var for var, var_type in FEATURE_VARIABLES.items() if var_type == 'macro')
 
def generate_plot(df, var_names, bank_vars, macro_vars):
    num_plots = len(var_names)
    num_cols = 3
    num_rows = math.ceil(num_plots / num_cols)
 
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4), sharex=True)
    axes_flat = axes.flatten()
 
    for i, var_name in enumerate(var_names):
        ax = axes_flat[i]
        pub_name = PUBLICATION_NAMES.get(var_name, var_name)
        if var_name not in df.columns:
            print(f"Warning: Variable '{var_name}' not found in df_final. Skipping plot.")
            ax.set_title(f"{pub_name} (Not Found)")
            ax.text(0.5, 0.5, "Data N/A", ha="center", va="center", transform=ax.transAxes)
            continue
 
        if var_name in macro_vars:
            # Macro variable: plot directly (taking the first value per date)
            macro_data_ts = df.groupby('date')[var_name].mean()
            ax.plot(macro_data_ts.index, macro_data_ts.values, label=pub_name, color='purple')
            ax.set_title(f"{pub_name} (Macro)")
        elif var_name in bank_vars:
            # Bank-level variable: plot mean, median, and IQR
            grouped = df.groupby('date')[var_name]
            mean_ts = grouped.mean()
            median_ts = grouped.median()
            q1_ts = grouped.quantile(0.25)
            q3_ts = grouped.quantile(0.75)
 
            ax.plot(mean_ts.index, mean_ts.values, label='Mean', color='blue')
            ax.plot(median_ts.index, median_ts.values, label='Median', color='orange', linestyle='--')
            ax.fill_between(q1_ts.index, q1_ts.values, q3_ts.values, color='skyblue', alpha=0.4, label='IQR')
            ax.set_title(f"{pub_name} (Bank-Level)")
        else:
            # Fallback for variables not in either set
            print(f"Warning: Variable '{var_name}' not categorized as bank or macro. Skipping plot.")
            ax.set_title(f"{pub_name} (Unknown Type)")
            ax.text(0.5, 0.5, "Data N/A", ha="center", va="center", transform=ax.transAxes)
            continue
 
        ax.legend(fontsize='small')
        ax.tick_params(axis='x', rotation=45)
 
    for j in range(i + 1, len(axes_flat)): # Hide any unused subplots
        fig.delaxes(axes_flat[j])
 
    fig.suptitle("Target and Feature Variables Over Time", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    return fig, plt

# Plot Bank-Specific Variables
fig_bank_vars, plt_bank_vars = generate_plot(df_final, bank_vars_set, bank_vars_set, macro_vars_set)
fig_bank_vars.suptitle("Bank-Specific Variables Over Time", fontsize=16, y=1.02)
plt_bank_vars.show()
# Save the figure with date and time in front of the filename in the format YYYYMMDD_HHMM
import datetime
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M")
fig_bank_vars.savefig(f"plots/bank_variables_{timestamp}.png", bbox_inches='tight')

# Plot Macroeconomic Variables
fig_macro_vars, plt_macro_vars = generate_plot(df_final, macro_vars_set, bank_vars_set, macro_vars_set)
fig_macro_vars.suptitle("Macroeconomic Variables Over Time", fontsize=16, y=1.02)
plt_macro_vars.show()
# Save the figure with date and time in front of the filename in the format YYYYMMDD_HHMM
now = datetime.datetime.now() # Re-get current time to ensure unique timestamp if plots are fast
timestamp = now.strftime("%Y%m%d_%H%M")
fig_macro_vars.savefig(f"plots/macro_variables_{timestamp}.png", bbox_inches='tight')

# Clear the figures to prevent them from being displayed again if the script is run in an interactive environment
plt.close(fig_bank_vars)
plt.close(fig_macro_vars)



#endregion


# Make a correlation plot of the target and feature variables
print("\n--- Correlation Plot of Target and Feature Variables ---------------------------------")
# Calculate the correlation matrix for the target and feature variables
correlation_vars = list(TARGET_VARIABLES.keys()) + list(FEATURE_VARIABLES.keys())
correlation_matrix = df_final[correlation_vars].corr().rename(index=PUBLICATION_NAMES, columns=PUBLICATION_NAMES)
correlation_vars_renamed = [PUBLICATION_NAMES.get(v, v) for v in correlation_vars]

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_vars_renamed)), correlation_vars_renamed, rotation=45, ha='right')
plt.yticks(range(len(correlation_vars_renamed)), correlation_vars_renamed)
plt.title("Correlation Matrix of Target and Feature Variables")
plt.tight_layout()
plt.show()

# Plot the correlation matrix as a heatmap
# import seaborn as sns
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
# plt.title("Correlation Heatmap of Target and Feature Variables")
# plt.tight_layout()
# plt.show()

# Show a distribution of total assets
print("\n--- Distribution of Total Assets -------------------------------------------------------------")

# Filter data for the year 2022
df_2022 = df_final[df_final.index.get_level_values('date').year == 2022]
df_2022['total_assets'] = np.exp(df_2022['log_total_assets'])  # Convert log total assets back to total assets

# Plot the distribution of 'total_assets' for 2022
plt.figure(figsize=(10, 6))
sns.histplot(df_2022['total_assets'], kde=True, bins=50)
plt.title('Distribution of Total Assets in 2022')
plt.xlabel('Total Assets')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

df_final.index.get_level_values('id').nunique()
