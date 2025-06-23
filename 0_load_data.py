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



# --- Main Processing ---
# This section serves as the main execution block of the script.
# It calls the functions to load, merge, and calculate ratios,
# then saves the final processed data.

# Load data
fred = pd.read_parquet('data/fred/macro_data_processed.parquet')
fdic = pd.read_parquet('data/fdic/fdic_data_processed.parquet')
yahoo = pd.read_parquet('data/yahoo/yahoo.parquet')
print("Data loaded.")

# Merge initial datasets
print("Merging data...")
data_merged = fred.merge(fdic, on='date', how='left').merge(yahoo, on='date', how='left')
data_merged.set_index(['id', 'date'], inplace=True)
print("Data merged and index set.")

# Remove rows with nan ids
data_merged = data_merged[~data_merged.index.get_level_values('id').isna()]


# Save the processed data
print(f"\nSaving processed data to {OUTPUT_FILENAME}...")
data_merged.to_parquet(OUTPUT_FILENAME)
print(f"Data saved to {OUTPUT_FILENAME}.")
print("Script finished.")
