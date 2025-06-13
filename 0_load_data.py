"""
This script orchestrates the initial data loading, merging, and preprocessing steps
for the stress prediction project. It performs the following main operations:
1. Loads raw data from FRED, FDIC, and Yahoo sources (parquet files).
2. Merges these datasets based on the 'date' column.
3. Calculates a comprehensive set of financial ratios.
4. Saves the processed DataFrame (with merged data and calculated ratios) to a parquet file.

Note: More advanced cleaning, winsorization, and feature engineering (like lagging)
are handled by the RegressionDataPreparer class in subsequent scripts.
"""
import pandas as pd
import numpy as np

# --- Configuration ---\
OUTPUT_FILENAME = "data.parquet"

# --- Helper Functions ---

def calculate_financial_ratios(df):
    """
    Calculates various financial ratios and adds them as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with necessary columns
                           (e.g., total_deposits, total_assets, etc.).

    Returns:
        pd.DataFrame: DataFrame with added ratio columns.
    """
    print("Calculating financial ratios...")
    df2 = df.copy()

    # Calculate ratios
    df2['deposit_ratio'] = df2['total_deposits'] / df2['total_assets']
    df2['loan_to_deposit_ratio'] = df2['total_loans_and_leases'] / df2['total_deposits']
    df2['loan_to_asset_ratio'] = df2['total_loans_and_leases'] / df2['total_assets']
    df2['equity_to_asset_ratio'] = df2['total_equity'] / df2['total_assets']
    df2['trading_assets_ratio'] = df2['trading_assets'] / df2['total_assets']
    df2['net_interest_margin'] = (
        df2['interest_income'] - df2['interest_expense']
    ) / df2['total_assets']
    df2['roe'] = df2['net_income'] / df2['total_equity']
    df2['roa'] = df2['net_income'] / df2['total_assets']
    df2[         'net_income_to_assets'] =  df2[         'net_income']  / df2['total_assets']
    df2['net_interest_income_to_assets'] =  df2['net_interest_income']  / df2['total_assets']
    df2[    'interest_income_to_assets'] =  df2[    'interest_income']  / df2['total_assets']
    df2[    'interest_expense_to_assets'] = df2[    'interest_expense'] / df2['total_assets']
    df2['non_interest_income_to_assets'] =  df2['non_interest_income']  / df2['total_assets']
    df2['non_interest_expense_to_assets'] = df2['non_interest_expense'] / df2['total_assets']
    df2['net_charge_offs_to_loans_and_leases'] = df2['net_charge_offs'] / df2['total_loans_and_leases']
    df2['npl_ratio'] = df2['npl'] / df2['total_loans_and_leases']
    df2['charge_off_ratio'] = df2['total_charge_offs'] / df2['total_loans_and_leases']
    df2['allowance_for_loan_and_lease_losses_to_assets'] = (
        df2['allowance_for_loan_and_lease_losses'] / df2['total_assets'])
    df2['allowance_for_credit_losses_to_assets'] = (
        df2['allowance_for_credit_losses'] / df2['total_assets'])
    df2['provisions_for_credit_losses_to_assets'] = (
        df2['provisions_for_credit_losses'] / df2['total_assets'])
    df2['rwa_ratio'] = df2['total_rwa'] / df2['total_assets']
    df2['dep_small_3m_less_to_assets'] = df2['dep_small_3m_less'] / df2['total_assets']
    df2['dep_small_3m_1y_to_assets']   = df2['dep_small_3m_1y']   / df2['total_assets']
    df2['dep_small_1y_3y_to_assets']   = df2['dep_small_1y_3y']   / df2['total_assets']
    df2['dep_small_3y_more_to_assets'] = df2['dep_small_3y_more'] / df2['total_assets']
    df2['dep_large_3m_less_to_assets'] = df2['dep_large_3m_less'] / df2['total_assets']
    df2['dep_large_3m_1y_to_assets']   = df2['dep_large_3m_1y']   / df2['total_assets']
    df2['dep_large_1y_3y_to_assets']   = df2['dep_large_1y_3y']   / df2['total_assets']
    df2['dep_large_3y_more_to_assets'] = df2['dep_large_3y_more'] / df2['total_assets']


    # Log total assets
    df2['log_total_assets'] = np.log(df2['total_assets'].replace(0, np.nan).fillna(1e-9))

    print("Financial ratios calculated.")
    return df2

def load_and_merge_data():
    """
    Loads raw data from FRED, FDIC, and Yahoo parquet files,
    merges them on the 'date' column, and sets a MultiIndex ('id', 'date').

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    print("Loading data...")
    fred = pd.read_parquet('data/fred/macro_data_processed.parquet')
    fdic = pd.read_parquet('data/fdic/fdic_data_processed.parquet')
    yahoo = pd.read_parquet('data/yahoo/yahoo.parquet')
    print("Data loaded.")

    print("Merging data...")
    merged_df = fdic.merge(fred, on='date', how='left').merge(yahoo, on='date', how='left')
    merged_df.set_index(['id', 'date'], inplace=True)
    print("Data merged and index set.")
    return merged_df

# --- Main Processing ---
# This section serves as the main execution block of the script.
# It calls the functions to load, merge, and calculate ratios,
# then saves the final processed data.

# Load and merge initial datasets
data_merged = load_and_merge_data()

# Calculate financial ratios
data_merged = calculate_financial_ratios(data_merged)

# Save the processed data
print(f"\nSaving processed data to {OUTPUT_FILENAME}...")
data_merged.to_parquet(OUTPUT_FILENAME)
print(f"Data saved to {OUTPUT_FILENAME}.")
print("Script finished.")
