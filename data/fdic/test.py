import pandas as pd
import numpy as np
from pathlib import Path

def test_parquet_output():
    """
    Reads parquet files from the 'intermediate' directory and prints
    statistics regarding dimensions and data density.
    """
    data_dir = Path("data/fdic/intermediate")
    
    if not data_dir.exists():
        print(f"Directory '{data_dir}' does not exist. Run the processing script first.")
        return

    files = list(data_dir.glob("*.parquet"))
    # Delete from files code_descriptions.parquet
    files = [f for f in files if f.name != "code_descriptions.parquet"]
    
    if not files:
        print(f"No parquet files found in '{data_dir}'.")
        return

    # Header
    print(f"{'File Name':<40} | {'Banks':>8} | {'Fields':>8} | {'Missing Cells':>15} | {'Data Cells':>15}")
    print("-" * 95)

    for p_file in sorted(files):
        try:
            # Load data
            df = pd.read_parquet(p_file)
            
            # 1. Number of Banks
            # The processing script ensures IDRSSD is a column (via reset_index)
            if 'IDRSSD' in df.columns:
                num_banks = df['IDRSSD'].nunique()
            else:
                # Fallback if IDRSSD is missing for some reason
                num_banks = len(df)

            # 2. Number of Fields (Columns)
            num_fields = df.shape[1]

            # 3. Matrix Analysis (Missing vs Information)
            # Since the generator script forced .astype(str) to avoid PyArrow type errors,
            # null values were converted to the string "nan".
            # We count actual NaNs, string "nan" (case insensitive), "None", and empty strings.
            
            # Create mask for missing values
            # specific string check for 'nan' is crucial due to the previous string conversion
            missing_mask = df.isin(['nan', 'NaN', 'None', '']) | df.isna()
            
            num_missing = missing_mask.sum().sum()
            total_cells = df.size
            num_present = total_cells - num_missing

            # Output row
            print(f"{p_file.name:<40} | {num_banks:>8} | {num_fields:>8} | {num_missing:>15} | {num_present:>15}")


        except Exception as e:
            print(f"Error processing {p_file.name}: {e}")

if __name__ == "__main__":
    test_parquet_output()