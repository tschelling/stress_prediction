import pandas as pd
import os

# Define the base directory and target/horizon
base_dir = "models_and_results1"
target_variable = "interest_expense_to_assets"
horizon = 1
data_path = os.path.join(base_dir, f"target_{target_variable}", f"horizon_{horizon}", "data")

# List all parquet files in the specified directory
parquet_files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]

# Read each parquet file and assign it to a variable named after the filename (without extension)
for file_name in parquet_files:
    file_path = os.path.join(data_path, file_name)
    variable_name = os.path.splitext(file_name)[0]  # Get filename without extension
    globals()[variable_name] = pd.read_parquet(file_path)
    print(f"Loaded {file_name} into variable '{variable_name}'")

# Example of how to access the loaded dataframes:
# print(X_train_scaled.head())
# print(y_test.head())
