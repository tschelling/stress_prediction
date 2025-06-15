# --- Imports ---
import pandas as pd
import numpy as np
import zipfile
import glob
import os
import re
from datetime import datetime

# --- Global Configuration ---
DO_DEBUG = False  # If True, only the first 10 zip files will be processed.
# Regex for Split File Detection: Matches patterns like "...(1 of 3).txt"
# Groups: 1=Base Name, 2=Part Number, 3=Total Parts
SPLIT_FILE_PATTERN = re.compile(r"^(.*?)\s*\((\d+)\s+of\s+(\d+)\)\.txt$", re.IGNORECASE)

# --- Utility Functions ---

def extract_reporting_quarter(filename: str) -> str:
    """
    Extracts reporting quarter (e.g., "Q1 2023") from a filename.
    Searches for YYYYMMDD or MMDDYYYY patterns within any 8-digit sequences.
    """
    potential_dates = re.findall(r'\d{8}', filename)  # Find all 8-digit sequences
    
    for date_str in potential_dates:
        # Try parsing as YYYYMMDD
        try:
            dt = datetime.strptime(date_str, '%Y%m%d')
            # Validate year to ensure it's within a reasonable range
            if 1980 <= dt.year <= 2050: 
                quarter = (dt.month - 1) // 3 + 1
                return f"Q{quarter} {dt.year}"
        except ValueError:
            pass  # Not YYYYMMDD or invalid date

        # Try parsing as MMDDYYYY
        try:
            dt = datetime.strptime(date_str, '%m%d%Y')
            if 1980 <= dt.year <= 2050:
                quarter = (dt.month - 1) // 3 + 1
                return f"Q{quarter} {dt.year}"
        except ValueError:
            pass  # Not MMDDYYYY or invalid date
            
    return "Unknown" # Return "Unknown" if no valid date pattern is found

def log_df_metrics(df_to_log: pd.DataFrame | None, file_identifier_for_log: str, debug_data_list: list):
    """
    Calculates and appends debug metrics for a given DataFrame to the debug_data_list.
    Assumes df_to_log has IDRSSD as index if it's valid and processed.
    
    Args:
        df_to_log: The DataFrame to analyze. Can be None or empty.
        file_identifier_for_log: Name of the file or derived key for logging.
        debug_data_list: List to append the metrics dictionary to.
    """
    quarter = extract_reporting_quarter(file_identifier_for_log)
    
    num_cells = 0
    num_mv = 0
    num_banks_val = 0
    non_numeric_idrssd_val = 0

    if df_to_log is None or df_to_log.empty:
        # Metrics remain 0 for empty or None DataFrame
        # num_mv could be df_to_log.isnull().sum().sum() if an empty df with columns is passed
        if df_to_log is not None: # handles df that is empty but not None
             num_mv = df_to_log.isnull().sum().sum()
    else:
        num_mv = df_to_log.isnull().sum().sum()
        num_cells = df_to_log.size
        # Check if IDRSSD is the index (standard for processed DFs)
        if df_to_log.index.name == 'IDRSSD':
            idrssd_series_from_index = pd.Series(df_to_log.index.astype(str))
            num_banks_val = idrssd_series_from_index.nunique()
            # Count IDRSSDs that are not purely numeric after stripping whitespace
            non_numeric_idrssd_val = idrssd_series_from_index.str.strip().apply(
                lambda x: not x.isdigit() if x else False # if x ensures empty strings are not an issue
            ).sum()
        else:
            # Fallback if IDRSSD is not the index (e.g., set_index failed)
            print(f"    - Warning (Debug Logging): IDRSSD not index for '{file_identifier_for_log}' when logging metrics.")
            if 'IDRSSD' in df_to_log.columns: # Check if IDRSSD is a column
                try:
                    idrssd_col_series = df_to_log['IDRSSD'].astype(str).str.strip()
                    # Filter out empty strings before counting
                    valid_idrssd_col_series = idrssd_col_series[idrssd_col_series != '']
                    if not valid_idrssd_col_series.empty:
                        num_banks_val = valid_idrssd_col_series.nunique()
                        non_numeric_idrssd_val = valid_idrssd_col_series.apply(
                            lambda x: not x.isdigit() if x else False
                        ).sum()
                except Exception as e:
                    print(f"    - Error processing 'IDRSSD' column for debug log on '{file_identifier_for_log}': {e}")

    debug_data_list.append({
        'text_file_name': file_identifier_for_log,
        'reporting_quarter': quarter,
        'num_banks': num_banks_val,
        'num_non_numeric_idrssd': non_numeric_idrssd_val,
        'num_cells': num_cells,
        'num_missing_values_total': num_mv,
        'perc_missing_values': np.round((num_mv / num_cells * 100),1) if num_cells > 0 else 0,
    })

# --- Core Processing Functions ---

def _read_header_metadata(file_in_zip, base_filename: str, code_metadata_map: dict):
    """
    Reads the first two rows of a file for code/metadata mapping.
    Updates the code_metadata_map directly.
    Ensures the file stream is reset to its original position.
    """
    try:
        # Read first two lines as strings to capture codes and descriptions
        header_meta_df = pd.read_csv(file_in_zip, sep='\t', header=None, nrows=2, encoding='utf-8', low_memory=False, dtype=str)
        if header_meta_df.shape[0] == 2: # Ensure two rows were actually read
            codes = header_meta_df.iloc[0].values
            descriptions = header_meta_df.iloc[1].values
            for code, desc in zip(codes, descriptions):
                code_str = str(code).strip().strip('"')
                desc_str = str(desc).strip().strip('"')
                # Add to map if code is valid, not 'IDRSSD', not already mapped,
                # and description is not purely numeric (heuristic for actual descriptions)
                if code_str and code_str != 'IDRSSD' and code_str not in code_metadata_map and pd.notna(desc) and not desc_str.isdigit():
                    code_metadata_map[code_str] = desc_str
    except Exception as e:
        print(f"    - Warning: Could not read header/metadata rows from '{base_filename}': {e}")
    finally:
        file_in_zip.seek(0) # IMPORTANT: Reset stream for subsequent data read

def _read_main_data(file_in_zip, base_filename: str) -> pd.DataFrame | None:
    """
    Reads the main data from the file stream (after header/metadata).
    Handles common encoding and type errors during CSV parsing.
    Returns a DataFrame or None if a critical error occurs.
    """
    # Default parameters for reading the tab-separated data
    read_params = {
        'sep': '\t', 
        'low_memory': False, 
        'dtype': {'IDRSSD': str}, # Initially read IDRSSD as string
        'encoding': 'utf-8', 
        'header': 0, 
        'skiprows': [1] # Skip the second row (metadata descriptions, header is row 0)
    }
    try:
        df = pd.read_csv(file_in_zip, **read_params)
    except UnicodeDecodeError:
        # Fallback to 'latin1' if UTF-8 fails
        print(f"    - Warning: UTF-8 decoding failed for '{base_filename}'. Trying 'latin1'.")
        read_params['encoding'] = 'latin1'
        file_in_zip.seek(0) # Reset stream before retrying
        df = pd.read_csv(file_in_zip, **read_params)
    except ValueError as ve:
        # Handle cases where 'IDRSSD' might have mixed types or unconvertible NAs
        if 'Integer column has NA values' in str(ve) or 'cannot safely convert' in str(ve):
            print(f"    - Warning: Potential mixed types or NAs in '{base_filename}'. Retrying with object dtype for all columns.")
            read_params.pop('dtype', None) # Remove specific dtype for IDRSSD
            file_in_zip.seek(0) # Reset stream
            df = pd.read_csv(file_in_zip, **read_params, dtype=object) # Read all columns as object
        else:
            # Re-raise other ValueErrors
            print(f"    - Error: ValueError during data read for '{base_filename}': {ve}")
            return None 
    except Exception as e:
        print(f"    - Error: Unexpected error during data read for '{base_filename}': {e}")
        return None
    return df

def _clean_and_prepare_dataframe(df: pd.DataFrame, base_filename: str) -> pd.DataFrame | None:
    """
    Performs basic validation, cleaning (IDRSSD handling, NaNs, duplicates), 
    and sets 'IDRSSD' as the DataFrame index.
    
    Args:
        df: The raw DataFrame loaded from the file.
        base_filename: The name of the file for logging purposes.
        
    Returns:
        A cleaned DataFrame with 'IDRSSD' as index, or None if critical issues are found
        or the DataFrame becomes empty after essential cleaning.
    """
    # Initial check if DataFrame is empty (e.g., file had headers but no data)
    if df.empty:
        print(f"    - Warning: File '{base_filename}' loaded as empty DataFrame initially.")
        return None 

    # Clean column names (strip whitespace and quotes)
    df.columns = [str(col).strip().strip('"') for col in df.columns]

    # Check for 'IDRSSD' column presence
    if 'IDRSSD' not in df.columns:
        print(f"    - Warning: 'IDRSSD' column not found in '{base_filename}'.")
        # Return the DataFrame as is; logging function will note absence of IDRSSD-related metrics
        return df 

    # Standardize 'IDRSSD': convert to string, drop NA IDRSSDs, strip whitespace, remove empty strings
    df['IDRSSD'] = df['IDRSSD'].astype(str)
    df = df.dropna(subset=['IDRSSD'])
    df['IDRSSD'] = df['IDRSSD'].str.strip()
    df = df[df['IDRSSD'] != ''] # Filter out rows where IDRSSD became an empty string

    # If DataFrame is empty after IDRSSD cleaning, it's not useful
    if df.empty:
        print(f"    - Warning: File '{base_filename}' has no valid IDRSSD values after cleaning.")
        return None

    # Drop rows where all non-IDRSSD columns are NaN
    non_idrssd_cols = df.columns.difference(['IDRSSD'])
    if not non_idrssd_cols.empty: # Check if there are any non-IDRSSD columns
        df = df.dropna(subset=non_idrssd_cols, how='all')
    
    if df.empty: # If empty after dropping all-NaN data rows
        print(f"    - Warning: File '{base_filename}' has no valid data after removing empty data rows.")
        return None

    # Handle duplicate 'IDRSSD' values by keeping the first occurrence
    if df['IDRSSD'].duplicated().any():
        print(f"      - Warning: Duplicate 'IDRSSD' values found in '{base_filename}'. Keeping first occurrence.")
        df = df.drop_duplicates(subset='IDRSSD', keep='first')

    # Set 'IDRSSD' as the index
    try:
        df = df.set_index('IDRSSD')
    except KeyError: # Should not happen if 'IDRSSD' column check passed and was not dropped
        print(f"    - Error: Could not set 'IDRSSD' as index for '{base_filename}'.")
        # Return DataFrame with 'IDRSSD' as column if setting index fails
        return df 
    
    return df

def load_single_text_file(file_in_zip, base_filename: str, code_metadata_map: dict, debug_data_list: list, shape_log_list: list) -> pd.DataFrame | None:
    """
    Loads, processes, and cleans a single text file from an open zip stream.
    Updates metadata, logs metrics, and returns the processed DataFrame (indexed by IDRSSD) or None.
    
    Args:
        file_in_zip: Open file object from the zip archive.
        base_filename: Name of the file being processed.
        code_metadata_map: Dictionary to update with code-description mappings.
        debug_data_list: List to append debug metrics to.
        shape_log_list: List to append raw file shape information to.
        
    Returns:
        A processed pandas DataFrame with 'IDRSSD' as index, or None if processing fails or file is invalid.
    """
    # Step 1: Read header/metadata (updates code_metadata_map)
    _read_header_metadata(file_in_zip, base_filename, code_metadata_map)
    
    # Step 2: Read the main data content
    df = _read_main_data(file_in_zip, base_filename)

    # Log the raw shape of the DataFrame immediately after reading
    if df is not None:
        shape_log_list.append({
            'text_file_name': base_filename,
            'rows': df.shape[0],
            'cols': df.shape[1]
        })
    else: # Handles cases where _read_main_data returns None (e.g., read error)
        shape_log_list.append({
            'text_file_name': base_filename, 'rows': 0, 'cols': 0
        })
    
    # If reading main data fails or returns an empty DataFrame early
    if df is None:
        log_df_metrics(None, base_filename, debug_data_list) # Log as None/empty
        return None
    if df.empty: # df might be empty but not None if only headers were present
        log_df_metrics(df, base_filename, debug_data_list) # Log empty DataFrame
        return None

    # Step 3: Clean and prepare the DataFrame (sets IDRSSD as index if successful)
    # Pass a copy to _clean_and_prepare_dataframe to avoid side effects if cleaning is partial
    cleaned_df = _clean_and_prepare_dataframe(df.copy(), base_filename) 

    # Step 4: Log metrics based on the state of cleaned_df (or df if cleaning_df is None)
    # If cleaned_df is None, it means it was considered invalid/empty during cleaning.
    # If 'IDRSSD' is not the index of cleaned_df (and cleaned_df is not None), set_index likely failed.
    log_df_metrics(cleaned_df if cleaned_df is not None else df, base_filename, debug_data_list)
    
    # Ensure only DataFrames that were successfully processed AND indexed are returned
    if cleaned_df is None or cleaned_df.index.name != 'IDRSSD':
        if cleaned_df is not None and cleaned_df.index.name != 'IDRSSD':
             # This specific check identifies cases where cleaning occurred but indexing failed
             print(f"    - Post-cleaning: 'IDRSSD' index not properly set for '{base_filename}'. File will be skipped for merging.")
        return None 

    return cleaned_df


def handle_split_files_combination(split_file_groups: dict, dfs_in_zip: dict, debug_data_list: list):
    """
    Combines completed split file groups. Adds combined DataFrames (or orphan parts) to dfs_in_zip.
    Modifies dfs_in_zip and clears processed groups from split_file_groups.
    
    Args:
        split_file_groups: Dict mapping (base_name, total_parts) to {part_num: DataFrame}.
        dfs_in_zip: Dict to store final DataFrames (key: unique file key, value: DataFrame).
        debug_data_list: List for logging metrics of combined DataFrames.
    """
    print("\n  - Checking for complete split file groups...")
    # Iterate over a copy of keys as the dictionary might be modified during iteration
    for group_key in list(split_file_groups.keys()):
        base_name_part, total_parts = group_key
        parts_dict = split_file_groups[group_key]

        # Check if all parts for this group have been collected
        if len(parts_dict) == total_parts and all(i in parts_dict for i in range(1, total_parts + 1)):
            print(f"    - Found complete group: '{base_name_part}' ({total_parts} parts). Concatenating horizontally...")
            try:
                # Sort parts by part number to ensure correct concatenation order
                sorted_parts = [parts_dict[i] for i in range(1, total_parts + 1)]
                
                # Start with the first part as the base for combining
                combined_df = sorted_parts[0]
                for i in range(1, total_parts):
                    next_part_df = sorted_parts[i]
                    # Identify and drop overlapping columns from the part being added (next_part_df)
                    overlapping_cols = combined_df.columns.intersection(next_part_df.columns)
                    if len(overlapping_cols) > 0:
                        print(f"      - Warning: Overlapping columns in part {i+1} of '{base_name_part}': {list(overlapping_cols)}. Dropping from part {i+1}.")
                        next_part_df = next_part_df.drop(columns=overlapping_cols)
                    
                    # Concatenate horizontally (axis=1). Pandas aligns on index by default.
                    # It's crucial that all parts share the same IDRSSD index structure.
                    if not combined_df.index.equals(next_part_df.index):
                        # This is a potential issue if indices don't align perfectly (e.g. different sets of banks)
                        # pd.concat with axis=1 will perform an outer join on indices.
                        print(f"      - Warning: Indices for part {i+1} of '{base_name_part}' may not perfectly align with previous parts. Review merged output carefully.")
                    combined_df = pd.concat([combined_df, next_part_df], axis=1)

                # Generate a unique key for the newly combined DataFrame
                combined_key = f"{base_name_part}_Combined_{total_parts}parts"
                original_key = combined_key
                counter = 1
                while combined_key in dfs_in_zip: # Ensure key uniqueness
                     combined_key = f"{original_key}_{counter}"
                     counter += 1
                if combined_key != original_key:
                      print(f"      - Warning: Potential key conflict for combined group '{original_key}'. Using '{combined_key}'.")

                log_df_metrics(combined_df, combined_key, debug_data_list) # Log metrics for the combined DataFrame
                dfs_in_zip[combined_key] = combined_df # Store the combined DataFrame
                print(f"      - Combined group '{base_name_part}' shape: {combined_df.shape}. Stored as '{combined_key}'.")
                del split_file_groups[group_key] # Remove the processed group

            except Exception as e:
                 print(f"      - Error during concatenation for group '{group_key}': {e}. Skipping combination for this group.")
        else:
            # Handle incomplete groups: move their parts to dfs_in_zip as orphans
            print(f"    - Incomplete group found: '{base_name_part}'. Expected {total_parts} parts, found {len(parts_dict)}: {sorted(list(parts_dict.keys()))}.")
            for part_num, df_part in parts_dict.items():
                # These parts were already logged when first processed.
                # Here, they are just being stored with a more descriptive key indicating they are orphans.
                part_key = f"{base_name_part}_Part_{part_num}_of_{total_parts}"
                original_key = part_key
                counter = 1
                while part_key in dfs_in_zip: # Ensure unique key
                    part_key = f"{original_key}_{counter}"
                    counter +=1
                if part_key != original_key:
                     print(f"      - Warning: Potential key conflict for orphan part '{original_key}'. Using '{part_key}'.")
                dfs_in_zip[part_key] = df_part # df_part is already indexed and logged
                print(f"      - Storing incomplete part {part_num} of '{base_name_part}' as '{part_key}'. Shape: {df_part.shape}")
            del split_file_groups[group_key] # Remove the processed incomplete group


def merge_zip_dataframes(dfs_in_zip: dict) -> pd.DataFrame | None:
    """
    Merges all DataFrames in dfs_in_zip based on their common 'IDRSSD' index.
    
    Args:
        dfs_in_zip: Dictionary of DataFrames to merge, with 'IDRSSD' as index.
        
    Returns:
        A single merged DataFrame with 'IDRSSD' reset as a column, or None if no data or error.
    """
    if not dfs_in_zip:
        print("  - No mergeable DataFrames were loaded or derived for this zip.")
        return None

    print(f"\n  - Merging {len(dfs_in_zip)} DataFrames from the current zip...")
    
    # Strategy for selecting the base DataFrame for the merge:
    # 1. Prefer files resulting from a combination of split parts.
    # 2. Fallback to predefined preferred key name parts (e.g., 'ENT', 'POR').
    # 3. Final fallback: the DataFrame with the most cells (rows * columns).
    base_key = None
    combined_keys = [k for k in dfs_in_zip.keys() if "_Combined_" in k]
    if combined_keys:
         base_key = combined_keys[0] 
    
    if not base_key: 
        potential_base_keys = list(dfs_in_zip.keys())
        preferred_key_parts = ['ENT', 'POR', 'RC', 'RCA'] 
        for p_key_part in preferred_key_parts:
            for actual_key in potential_base_keys:
                 # Check if key starts with or contains the preferred part as a whole word/segment
                 if actual_key.startswith(p_key_part) or f" {p_key_part}" in actual_key or f"_{p_key_part}" in actual_key:
                      base_key = actual_key
                      break
            if base_key: break

    if not base_key and dfs_in_zip: 
        base_key = max(dfs_in_zip, key=lambda k: dfs_in_zip[k].shape[0] * dfs_in_zip[k].shape[1])

    if not base_key: 
         print("    - Error: Could not determine a base DataFrame for merging (this should not happen if dfs_in_zip is populated).")
         return None

    try:
        # Start with the chosen base DataFrame
        merged_df = dfs_in_zip.pop(base_key)
        print(f"    - Starting merge with base DataFrame: '{base_key}' (Shape: {merged_df.shape})")
        
        # Iteratively merge remaining DataFrames
        for key, df_to_merge in dfs_in_zip.items():
            print(f"    - Merging with: '{key}' (Shape: {df_to_merge.shape})...")
            # Ensure the DataFrame to merge has data columns (besides the index)
            if df_to_merge.shape[1] > 0: 
                merged_df = pd.merge(
                    merged_df, df_to_merge, 
                    left_index=True, right_index=True, # Merge on 'IDRSSD' index
                    how='outer', # Use outer join to keep all IDRSSDs
                    suffixes=('', f'_{key}') # Add suffix to overlapping column names from the right DataFrame
                )
                print(f"      - Current merged shape: {merged_df.shape}")
            else:
                print(f"      - Skipping merge for '{key}' as it has no data columns (only index).")
        
        merged_df = merged_df.reset_index() # Convert 'IDRSSD' index back to a column
        print(f"  - Merge complete for this zip. Final shape including IDRSSD column: {merged_df.shape}")
        return merged_df
    except Exception as e:
         print(f"    - Error during merge process: {e}")
         return None


def save_to_parquet(df: pd.DataFrame | None, zip_file_path: str):
    """
    Saves the given DataFrame to a Parquet file.
    The output filename is derived from the original zip file's name.
    Creates the 'parquet/' output directory if it doesn't exist.
    
    Args:
        df: The DataFrame to save. If None or empty, the function will not save.
        zip_file_path: Path to the original zip file, used for naming the output.
    """
    if df is None or df.empty:
        # This case is usually handled by the caller, but added as a safeguard.
        # print("\n  - No data to save (DataFrame is empty or None passed to save_to_parquet).")
        return

    base_zip_name = os.path.basename(zip_file_path)
    output_filename_base, _ = os.path.splitext(base_zip_name)
    output_dir = "parquet" # Define the output directory name
    
    # Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"  - Error: Could not create directory '{output_dir}': {e}. Cannot save Parquet file.")
        return
        
    output_filename = os.path.join(output_dir, f"{output_filename_base}.parquet")
    
    print(f"\n  - Saving merged data for '{zip_file_path}' to: '{output_filename}'")
    try:
        # Ensure 'IDRSSD' column is string type before saving, if it exists
        if 'IDRSSD' in df.columns:
            df['IDRSSD'] = df['IDRSSD'].astype(str)
        
        # Attempt to save using pyarrow engine first
        df.to_parquet(output_filename, index=False, engine='pyarrow')
        print(f"  - Successfully saved '{output_filename}' using pyarrow engine.")
    except ImportError: # If pyarrow is not installed
         print("    - Pyarrow engine not found. Attempting to use 'fastparquet' engine.")
         try:
             if 'IDRSSD' in df.columns: # Re-ensure IDRSSD type for fastparquet too
                df['IDRSSD'] = df['IDRSSD'].astype(str)
             df.to_parquet(output_filename, index=False, engine='fastparquet')
             print(f"  - Successfully saved '{output_filename}' using fastparquet engine.")
         except ImportError: # If fastparquet is also not installed
              print("    - Error: Neither 'pyarrow' nor 'fastparquet' engine found. Parquet file cannot be saved.")
         except Exception as e_fp: # Other errors with fastparquet
              print(f"    - Error saving Parquet file '{output_filename}' with fastparquet: {e_fp}")
    except Exception as e_pa: # Other errors with pyarrow
        print(f"  - Error saving Parquet file '{output_filename}' with pyarrow: {e_pa}")

def process_zip_archive(zip_file_path: str, code_metadata_map: dict, debug_data_list: list, shape_log_list: list):
    """
    Processes a single zip archive: extracts text files, cleans data, combines split files,
    merges all data for the zip, and saves the result to Parquet.
    Updates code_metadata_map, debug_data_list, and shape_log_list in place.
    """
    print(f"\n--- Processing Zip File: {zip_file_path} ---")
    # Dictionaries to hold DataFrames extracted from the current zip file
    dfs_in_zip = {}          # Stores non-split or combined DataFrames ready for final merge
    split_file_groups = {} # Stores parts of split files: Key (base_name, total_parts), Value {part_num: DataFrame}
    loaded_files_count = 0   # Counter for successfully loaded file parts/full files

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_namelist = zip_ref.namelist() # Get all file paths within the zip
            print(f"  Files found inside '{os.path.basename(zip_file_path)}': {[os.path.basename(f) for f in zip_namelist if f.lower().endswith('.txt')]}")

            # Loop through each file inside the current zip archive
            for internal_path in zip_namelist:
                base_filename = os.path.basename(internal_path)
                # Process only .txt files
                if not base_filename.lower().endswith('.txt'):
                    continue

                print(f"  - Attempting to load and process: '{base_filename}'")
                try:
                    with zip_ref.open(internal_path) as file_in_zip:
                        # loaded_df will be None if processing fails, otherwise DataFrame with IDRSSD as index
                        loaded_df = load_single_text_file(file_in_zip, base_filename, code_metadata_map, debug_data_list, shape_log_list)

                    if loaded_df is None: 
                        # load_single_text_file already prints reasons for skipping and logs metrics
                        print(f"    - Skipping '{base_filename}' due to issues during its loading/processing phase.")
                        continue # Move to the next file in the zip
                    
                    loaded_files_count += 1
                    
                    # Check if the loaded file is part of a split sequence
                    match = SPLIT_FILE_PATTERN.match(base_filename)
                    if match: # File is a part of a split
                        base_name_part, part_num_str, total_parts_str = match.groups()
                        part_num, total_parts = int(part_num_str), int(total_parts_str)
                        group_key = (base_name_part.strip(), total_parts) # Key for grouping parts

                        print(f"      Detected split file part: Base='{base_name_part.strip()}', Part={part_num}/{total_parts}. Shape: {loaded_df.shape}")
                        if group_key not in split_file_groups:
                            split_file_groups[group_key] = {} # Initialize group if first part
                        if part_num in split_file_groups[group_key]:
                             # This indicates a duplicate part number for the same split group
                             print(f"      - Warning: Duplicate part number {part_num} for group {group_key} (from '{base_filename}'). Overwriting previous part.")
                        split_file_groups[group_key][part_num] = loaded_df
                    else: # File is not part of a split (a standalone file)
                        # Ensure the standalone file has data columns (besides the index)
                        if loaded_df.shape[1] == 0: 
                            print(f"    - Skipping non-split file '{base_filename}' as it contains no data columns after processing.")
                            continue
                        
                        # Generate a unique key for storing this DataFrame in dfs_in_zip
                        file_key = os.path.splitext(base_filename)[0] # Use filename without extension as base for key
                        original_key = file_key
                        counter = 1
                        # Ensure key is unique, check against both non-split and base names of split groups
                        while file_key in dfs_in_zip or any(key[0] == file_key for key in split_file_groups.keys()):
                            file_key = f"{original_key}_{counter}"
                            counter += 1
                        if file_key != original_key: # If key was changed due to conflict
                             print(f"    - Warning: Key conflict for '{original_key}'. Using unique key '{file_key}'.")
                        dfs_in_zip[file_key] = loaded_df
                        print(f"      Stored non-split file as '{file_key}'. Shape: {loaded_df.shape}")

                except pd.errors.EmptyDataError: 
                    print(f"  - Warning: File '{base_filename}' is empty or unreadable inside zip (pandas EmptyDataError).")
                    log_df_metrics(pd.DataFrame(), base_filename, debug_data_list) # Log as empty DataFrame
                except Exception as e:
                    print(f"  - Error processing file '{base_filename}' from '{zip_file_path}': {e}")
        
        # After iterating through all files in the current zip archive:
        # 1. Combine any complete split file groups
        handle_split_files_combination(split_file_groups, dfs_in_zip, debug_data_list)
        
        # 2. Merge all resulting DataFrames (combined splits, orphan parts, non-splits)
        merged_df_for_zip = None
        if not dfs_in_zip: # If no DataFrames were successfully processed and stored
            print("  - No mergeable DataFrames were loaded or derived from this zip file.")
        else:
            merged_df_for_zip = merge_zip_dataframes(dfs_in_zip) 

        # 3. Save the final merged DataFrame for this zip to Parquet
        if merged_df_for_zip is not None and not merged_df_for_zip.empty:
            save_to_parquet(merged_df_for_zip, zip_file_path)
        elif loaded_files_count > 0 : # Files were loaded, but merge might have failed or resulted in empty
             print("\n  - No data to save for this zip (merge resulted in empty DataFrame or encountered errors).")
        else: # No valid files were loaded from this zip at all
             print("\n  - No valid FFIEC text files were found or successfully loaded in this zip file.")

    except zipfile.BadZipFile:
        print(f"Error: Failed to open '{zip_file_path}'. It might be corrupted or not a valid zip file.")
    except Exception as e: # Catch-all for other unexpected errors during zip processing
        print(f"An unexpected error occurred while processing zip file '{zip_file_path}': {e}")

# --- Main Execution ---
def main():
    """
    Main function to orchestrate the processing of all specified zip files.
    Initializes data containers, finds zip files, processes each, and saves aggregate logs.
    """
    # Initialize global containers for aggregated data across all zip files
    debug_data_list = []    # List to store dictionaries of debug metrics for each file/part
    code_metadata_map = {}  # Dictionary for code-to-description mapping
    shape_log_list = []     # List to store raw shape information for each text file

    print("Starting FFIEC data processing script...")
    print("Searching for zip files in 'zip/' directory...")
    # Assumes zip files are located in a subdirectory named 'zip' relative to the script
    zip_file_paths = glob.glob('zip/*.zip') 

    if not zip_file_paths:
        print("Error: No .zip files found in the 'zip/' directory. Please ensure files are present.")
        return # Exit if no zip files

    # Apply debug limit if DO_DEBUG is True
    if DO_DEBUG:
        original_zip_count = len(zip_file_paths)
        zip_file_paths = zip_file_paths[:10] # Process only the first 10 files
        print(f"DEBUG MODE ACTIVE: Processing only the first {len(zip_file_paths)} of {original_zip_count} zip files found.")
    
    print(f"Found {len(zip_file_paths)} zip files to process: {zip_file_paths}")

    # Process each identified zip file
    for zip_file_path in zip_file_paths:
        process_zip_archive(zip_file_path, code_metadata_map, debug_data_list, shape_log_list)

    # --- After processing all zip files, save the aggregated logs ---

    # Save Import Debug Log
    if debug_data_list:
        print("\n\n--- Import Debug Log Summary ---")
        debug_df = pd.DataFrame(debug_data_list)
        if not debug_df.empty:
            print(f"Total entries in debug log: {len(debug_df)}")
            print("First 5 entries of the debug log:")
            # Using to_string() for better console formatting of head()
            print(debug_df.head().to_string()) 
        else:
            print("Debug log DataFrame is empty (no metrics were collected).")
        try:
            debug_log_filename = "import_debug_log.csv"
            debug_df.to_csv(debug_log_filename, index=False)
            print(f"\nFull debug log saved to '{debug_log_filename}'")
        except Exception as e:
            print(f"\nError saving debug log file '{debug_log_filename}': {e}")
    else:
        print("\n\n--- No debug log data was collected during processing ---")

    # Save Code to Metadata Mapping
    if code_metadata_map:
        print("\n\n--- Code to Metadata Mapping Summary ---")
        metadata_df = pd.DataFrame(list(code_metadata_map.items()), columns=['Code', 'Metadata'])
        print(f"Total metadata mappings found: {len(metadata_df)}")
        if not metadata_df.empty:
            print("First 5 metadata mappings:")
            print(metadata_df.head().to_string())
        try:
            mapping_filename = "code_metadata_mapping.csv"
            metadata_df.to_csv(mapping_filename, index=False)
            print(f"\nFull metadata mapping saved to '{mapping_filename}'")
        except Exception as e:
            print(f"\nError saving metadata mapping file '{mapping_filename}': {e}")
    else:
        print("\n\n--- No code-metadata mapping was generated during processing ---")

    # Save Raw Text File Shape Log
    if shape_log_list:
        print("\n\n--- Raw Text File Shape Log ---")
        shape_df = pd.DataFrame(shape_log_list)
        print(f"Total text files logged for shape: {len(shape_df)}")
        print("First 5 entries of the shape log:")
        print(shape_df.head().to_string())
        shape_log_filename = "txt_import_shapes.csv"
        shape_df.to_csv(shape_log_filename, index=False)
        print(f"\nRaw text file shape log saved to '{shape_log_filename}'")
    else:
        print("\n\n--- No shape data was collected for text files ---")


    print("\n--- Finished processing all zip files ---")

if __name__ == '__main__':
    # This ensures main() is called only when the script is executed directly
    main()