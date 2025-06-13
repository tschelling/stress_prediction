import pandas as pd


# Added for Neural Network
import tensorflow as tf
from tensorflow import keras


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
