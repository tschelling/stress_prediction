# %% [markdown]
# # Configure

# %%
import pandas as pd
import numpy as np


# --- Configuration ---\
do_winsorize = False # Assuming this might be used later
lags = 1
reduce_banks = True
reduce_banks_to = 20

raw_variables = [
    'gdp_qoq', 'cpi_qoq', 'sp500_qoq', 'corp_bond_spread',
    'cons_sentiment_qoq', 'unemployment', 'household_delinq',
    'vix_qoq', 'spread_10y_3m',
    'total_assets', 'total_deposits', 'total_loans_and_leases',
    'trading_assets', 'net_interest_income', 'interest_income',
    'non_interest_income', 'interest_expense' 'tbill_3m', 'tbill_10y',
    'allowance_for_loan_and_leases_losses', 'allowance_for_credit_losses', 'allowance_for_loans_and_leases_losses',
    'provisions_for_credit_losses', 'net_charge_offs'
]

feature_variables = ['gdp_qoq', 'cpi_qoq', 'cons_sentiment_qoq', 'unemployment',
                     'household_delinq', 'tbill_3m', 'tbill_10y', 'spread_10y_3m',
                     'deposits_to_assets', 'loans_to_assets']

target_variable = 'net_interest_income_to_assets'

print("Configuration set.")

# %% [markdown]
# # Load and select data

# %%
# --- Load Data ---
print("Loading data...")
fred = pd.read_parquet('data/fred/macro_data_processed.parquet')
fdic = pd.read_parquet('data/fdic/fdic_data_processed.parquet')
yahoo = pd.read_parquet('data/yahoo/yahoo.parquet')

print("Data loaded.")

print("Merging data...")
data_selected = fdic.merge(fred, on='date', how='left').merge(yahoo, on='date', how='left')
data_selected.set_index(['id', 'date'], inplace=True)
print("Data merged and index set.")

# %% [markdown]
# # Transform

# %%
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
    df_processed = df.copy()

    # Calculate ratios
    df_processed['deposit_ratio'] = df_processed['total_deposits'] / df_processed['total_assets']
    df_processed['loan_to_deposit_ratio'] = df_processed['total_loans_and_leases'] / df_processed['total_deposits']
    df_processed['loan_to_asset_ratio'] = df_processed['total_loans_and_leases'] / df_processed['total_assets']
    df_processed['equity_to_asset_ratio'] = df_processed['total_equity'] / df_processed['total_assets']
    df_processed['trading_assets_ratio'] = df_processed['trading_assets'] / df_processed['total_assets']
    df_processed['net_interest_margin'] = (
        df_processed['interest_income'] - df_processed['interest_expense']
    ) / df_processed['total_assets']
    df_processed['roe'] = df_processed['net_income'] / df_processed['total_equity']
    df_processed['roa'] = df_processed['net_income'] / df_processed['total_assets']
    df_processed[         'net_income_to_assets'] =  df_processed[         'net_income']  / df_processed['total_assets']
    df_processed['net_interest_income_to_assets'] =  df_processed['net_interest_income']  / df_processed['total_assets']
    df_processed[    'interest_income_to_assets'] =  df_processed[    'interest_income']  / df_processed['total_assets']
    df_processed[    'interest_expense_to_assets'] = df_processed[    'interest_expense'] / df_processed['total_assets']
    df_processed['non_interest_income_to_assets'] =  df_processed['non_interest_income']  / df_processed['total_assets']
    df_processed['non_interest_expense_to_assets'] = df_processed['non_interest_expense'] / df_processed['total_assets']
    df_processed['net_charge_offs_to_loans_and_leases'] = df_processed['net_charge_offs'] / df_processed['total_loans_and_leases']
    df_processed['npl_ratio'] = df_processed['npl'] / df_processed['total_loans_and_leases']
    df_processed['charge_off_ratio'] = df_processed['total_charge_offs'] / df_processed['total_loans_and_leases']
    df_processed['allowance_for_loan_and_lease_losses_to_assets'] = (
        df_processed['allowance_for_loan_and_lease_losses'] / df_processed['total_assets'])
    df_processed['allowance_for_credit_losses_to_assets'] = (
        df_processed['allowance_for_credit_losses'] / df_processed['total_assets'])
    df_processed['provisions_for_credit_losses_to_assets'] = (
        df_processed['provisions_for_credit_losses'] / df_processed['total_assets'])
    df_processed['rwa_ratio'] = df_processed['total_rwa'] / df_processed['total_assets']

    # Log total assets
    df_processed['log_total_assets'] = np.log(df_processed['total_assets'].replace(0, np.nan).fillna(1e-9))

    ratio_cols = [
        'deposit_ratio', 'loan_to_deposit_ratio', 'loan_to_asset_ratio',
        'equity_to_asset_ratio', 'trading_assets_ratio', 'net_interest_margin',
        'roe', 'roa', 'npl_ratio', 'charge_off_ratio', 'rwa_ratio', 'log_total_assets'
    ]
    for col in ratio_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)

    print("Financial ratios calculated.")
    return df_processed

data_selected = calculate_financial_ratios(data_selected)

# %% [markdown]
# # Clean the data

# %%
print("Cleaning data...")
data_cleaned = data_selected.copy()
initial_rows = len(data_cleaned)
cleaning_tracker = {}
print(f"Initial number of rows: {initial_rows}")

# Delete rows with zero total assets
rows_before = len(data_cleaned)
data_cleaned = data_cleaned[data_cleaned['log_total_assets'] > -np.inf]
rows_deleted = rows_before - len(data_cleaned)
cleaning_tracker['cleaning_ta_zero'] = rows_deleted
print(f"Deleted {rows_deleted} rows with zero or negative total assets.")

# Delete roa values smaller than -1 and greater than 1
rows_before = len(data_cleaned)
data_cleaned = data_cleaned[(data_cleaned['roa'] > -1) & (data_cleaned['roa'] < 1)]
rows_deleted = rows_before - len(data_cleaned)
cleaning_tracker['cleaning_roa_invalid'] = rows_deleted
print(f"Deleted {rows_deleted} rows with ROA < -1 or > 1.")

# Delete rows with loans to total assets ratio greater than or equal to 1
rows_before = len(data_cleaned)
data_cleaned = data_cleaned[data_cleaned['loan_to_asset_ratio'] < 1]
rows_deleted = rows_before - len(data_cleaned)
cleaning_tracker['cleaning_l2ta_ge_1'] = rows_deleted
print(f"Deleted {rows_deleted} rows with loans to total assets ratio >= 1.")

# Delete rows with trading assets to total assets ratio < 0 or >= 1
rows_before = len(data_cleaned)
data_cleaned = data_cleaned[(data_cleaned['trading_assets_ratio'] < 1) & (data_cleaned['trading_assets_ratio'] >= 0)]
rows_deleted = rows_before - len(data_cleaned)
cleaning_tracker['cleaning_trada_invalid'] = rows_deleted
print(f"Deleted {rows_deleted} rows with trading assets to total assets ratio < 0 or >= 1.")

total_deleted = sum(cleaning_tracker.values())
print(f"\nTotal rows deleted: {total_deleted}")
print(f"Remaining rows: {len(data_cleaned)}")
print(f"\nShape after cleaning: {data_cleaned.shape}")
print("Data cleaning complete.")

# %% [markdown]
# # Winsorize

# %%
def winsorize_dataframe(df, vars_to_winsorize, do_winsorize_flag=False, lower_percentile=0.02, upper_percentile=0.98):
    """
    Winsorizes specified columns of a DataFrame.
    """
    df_out = df.copy(deep=True)

    if not do_winsorize_flag:
        print("Winsorization skipped as per do_winsorize_flag.")
        return df_out

    print(f"Winsorizing columns at [{lower_percentile}, {upper_percentile}] percentiles...")
    for col in vars_to_winsorize:
        if col in df_out.columns:
            if pd.api.types.is_numeric_dtype(df_out[col]):
                print(f" - Winsorizing {col}")
                lower_bound = df_out[col].quantile(lower_percentile)
                upper_bound = df_out[col].quantile(upper_percentile)
                df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)
            else:
                print(f" - Skipping non-numeric column {col}")
        else:
            print(f" - Skipping column {col} (not found in DataFrame)")
    print("Winsorization complete.")
    return df_out

accounting_columns = [col for col in data_cleaned.columns if 'ratio' in col or 'assets' in col or 'roa' in col]
data_winsorized = winsorize_dataframe(data_cleaned, accounting_columns, do_winsorize_flag=do_winsorize, lower_percentile=0.02, upper_percentile=0.98)
# The original notebook had a print statement here even if winsorization was skipped by do_winsorize.
# I'm keeping it consistent, but you might want to move it inside the if do_winsorize_flag block in the function.
print(f"Selected columns for potential winsorization: {accounting_columns}")


# %% [markdown]
# # Lag the data

# %%
data_processed = data_winsorized.copy()
original_cols = list(data_processed.columns)

print("Starting lag generation...")
nr_generated_lags = 0
for col in original_cols:
     for lag in range(1, lags + 1):
          data_processed[f'{col}_lag{lag}'] = data_processed.groupby(level='id')[col].shift(lag)
          nr_generated_lags += 1

print(f"Generated {nr_generated_lags} lagged features.")
print("Lag generation complete.")

# %% [markdown]
# # Save the data

# %%
print("Saving processed data...")
data_processed.to_parquet("data.parquet")
print("Data saved to data.parquet.")

print("Script finished.")