import pandas as pd


import regression_data_preparer
import importlib
importlib.reload(regression_data_preparer)  # Ensure the latest version of PanelDataPreparer is used
from regression_data_preparer import RegressionDataPreparer

TARGET_VARIABLE = 'interest_income_to_assets'
FEATURE_VARIABLES = ['gdp_qoq', 'deposit_ratio', 'loan_to_asset_ratio', 'log_total_assets', 'cpi_qoq', 'unemployment', 
                     'household_delinq', 'tbill_3m', 'tbill_10y', 'spread_10y_3m', 'sp500_qoq',
                     'corp_bond_spread', 'vix_qoq', 'is_structural_break',
                     # 'dep_small_3m_less_to_assets',
                     # 'dep_small_3m_1y_to_assets',
                     # 'dep_small_1y_3y_to_assets',
                     # 'dep_small_3y_more_to_assets',
                     # 'dep_large_3m_less_to_assets',
                     # 'dep_large_3m_1y_to_assets', 
                     # 'dep_large_1y_3y_to_assets',
                     # 'dep_large_3y_more_to_assets'
                     ] # Added more macro vars
FORECAST_HORIZONS = list(range(1, 2))


c = {
    'TARGET_VARIABLE': TARGET_VARIABLE,                 
    'FEATURE_VARIABLES': FEATURE_VARIABLES,             
    'INCLUDE_TIME_FE': True,                           
    'INCLUDE_BANK_FE': False,    
    'INCLUDE_STRUCTURAL_BREAK_DUMMY': True,               
    'OUTLIER_THRESHOLD_TARGET': 3.0,                    
    'MIN_OBS_PER_BANK': 12,             
    'CORRECT_STRUCTURAL_BREAKS_TOTAL_ASSETS': True,                
    'DATA_BEGIN': None, #'2017-01-01',                              
    'DATA_END': None,                                  
    'RESTRICT_TO_NUMBER_OF_BANKS': 20,                 
    'RESTRICT_TO_BANK_SIZE': None,                      
    'RESTRICT_TO_MINIMAL_DEPOSIT_RATIO': None,          
    'RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO': None,     
    'INCLUDE_AUTOREGRESSIVE_LAGS': True,                
    'NUMBER_OF_LAGS_TO_INCLUDE': 4,                     
    'TRAIN_TEST_SPLIT_DIMENSION': 'date',               
    'TEST_SPLIT': 0.25                                  # Takes a number or a date in the format 'YYYY-MM-DD'
}


print("--- Loading Data ---")
data_precleaning = pd.read_parquet('data.parquet')
print(f"Raw data loaded. Shape: {data_precleaning.shape}")


data_preparer = RegressionDataPreparer(initial_df=data_precleaning, config=c)


print("--- Missing value stats ----------------------------------------------------------------------------")
df0 = data_preparer.rectangularize_dataframe(data_preparer.df1_after_initial_cleaning).reset_index()
# Number of missing values in total assets
missing_total_assets = df0['log_total_assets'].isna().sum()
print(f"Number of missing values in log total assets: {missing_total_assets}")
# Number of banks in the full dataset
number_of_banks = df0['id'].nunique()
print(f"Number of banks in the dataset: {number_of_banks}")
# Number of banks with full data, i.e. no missing values in log total assets
banks_with_full_data = df0.groupby('id')['log_total_assets'].apply(lambda x: x.notna().all()).sum()
print(f"Number of banks with full data (no missing log total assets): {banks_with_full_data}")
# Number of banks with at least one missing value in log total assets
banks_with_missing_data = number_of_banks - banks_with_full_data
print(f"Number of banks with at least one missing value in log total assets: {banks_with_missing_data}")
print(f"Percentage of banks with full data: {banks_with_full_data / number_of_banks * 100:.2f}%")

# Number of banks removed with not enough observations
nr_banks_removed_not_enough_data = data_preparer.data_processing_stats['bank_removal_not_enough_data']['banks_before'] - data_preparer.data_processing_stats['bank_removal_not_enough_data']['banks_after']
print(f"Number of banks removed due to not enough observations: {nr_banks_removed_not_enough_data}")

# Missing values in total assets over time
missing_total_assets_over_time = df0.groupby('date')['log_total_assets'].apply(lambda x: x.isna().sum())
# Simple plot of missing values in total assets over time
missing_total_assets_over_time.plot(title='Missing values in log total assets over time', ylabel='Number of missing values', xlabel='Date')