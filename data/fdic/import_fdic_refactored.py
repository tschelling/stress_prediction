import glob
import re
import zipfile
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator, Pattern

import pandas as pd
import numpy as np

# --- Configuration ---

class Config:
    """Static configuration namespace."""
    ZIP_DIR = Path("data/fdic/zip")
    PARQUET_DIR = Path("data/fdic/parquet")
    DEBUG_MODE = True
    DEBUG_LIMIT = 1
    SPLIT_FILE_PATTERN = re.compile(r"^(.*?)\s*\((\d+)\s+of\s+(\d+)\)\.txt$", re.IGNORECASE)
    DATE_PATTERNS = ['%Y%m%d', '%m%d%Y']
    MERGE_PRIORITY_KEYS = ['ENT', 'POR', 'RC', 'RCA']

# --- Data Structures ---

@dataclass
class ProcessingMetrics:
    """Immutable container for file processing metrics."""
    text_file_name: str
    reporting_quarter: str
    num_banks: int = 0
    num_non_numeric_idrssd: int = 0
    num_cells: int = 0
    num_missing_values_total: int = 0
    perc_missing_values: float = 0.0

@dataclass
class FileShapeLog:
    """Immutable container for raw file dimensions."""
    text_file_name: str
    rows: int
    cols: int

@dataclass
class ProcessedFileResult:
    """Container for the result of processing a single file."""
    filename: str
    dataframe: Optional[pd.DataFrame]
    metrics: ProcessingMetrics
    shape_log: FileShapeLog
    metadata_map: Dict[str, str]
    is_split_part: bool = False
    split_base_name: Optional[str] = None
    split_part_num: int = 0
    split_total_parts: int = 0

# --- Pure Functional Logic ---

def extract_reporting_quarter(filename: str, date_patterns: List[str]) -> str:
    """
    Extracts the reporting quarter from a filename using regex and date parsing.
    Pure function.
    """
    potential_dates = re.findall(r'\d{8}', filename)
    
    for date_str in potential_dates:
        for pattern in date_patterns:
            try:
                dt = datetime.strptime(date_str, pattern)
                if 1980 <= dt.year <= 2050:
                    quarter = (dt.month - 1) // 3 + 1
                    return f"Q{quarter} {dt.year}"
            except ValueError:
                continue
    return "Unknown"

def calculate_metrics(df: Optional[pd.DataFrame], filename: str) -> ProcessingMetrics:
    """
    Calculates data quality metrics for a DataFrame.
    Pure function (does not modify df).
    """
    quarter = extract_reporting_quarter(filename, Config.DATE_PATTERNS)
    
    if df is None or df.empty:
        return ProcessingMetrics(
            text_file_name=filename,
            reporting_quarter=quarter,
            num_missing_values_total=0 if df is None else int(df.isnull().sum().sum())
        )

    num_mv = int(df.isnull().sum().sum())
    num_cells = int(df.size)
    perc_mv = np.round((num_mv / num_cells * 100), 1) if num_cells > 0 else 0.0
    
    num_banks = 0
    non_numeric_idrssd = 0

    # Handle IDRSSD counting based on index or column
    idrssd_series = None
    if df.index.name == 'IDRSSD':
        idrssd_series = pd.Series(df.index.astype(str))
    elif 'IDRSSD' in df.columns:
        idrssd_series = df['IDRSSD'].astype(str)

    if idrssd_series is not None:
        cleaned_series = idrssd_series.str.strip()
        valid_series = cleaned_series[cleaned_series != '']
        num_banks = valid_series.nunique()
        non_numeric_idrssd = valid_series.apply(lambda x: not x.isdigit()).sum()

    return ProcessingMetrics(
        text_file_name=filename,
        reporting_quarter=quarter,
        num_banks=num_banks,
        num_non_numeric_idrssd=non_numeric_idrssd,
        num_cells=num_cells,
        num_missing_values_total=num_mv,
        perc_missing_values=perc_mv
    )

def parse_header_metadata(file_obj) -> Dict[str, str]:
    """
    Reads first two rows of a file object to extract metadata.
    Side effect: Reads file stream.
    Returns: Dictionary of Code -> Description.
    """
    local_map = {}
    try:
        file_obj.seek(0)
        # Read only first two rows
        header_df = pd.read_csv(
            file_obj, sep='\t', header=None, nrows=2, 
            encoding='utf-8', low_memory=False, dtype=str
        )
        if header_df.shape[0] == 2:
            codes = header_df.iloc[0].values
            descriptions = header_df.iloc[1].values
            for code, desc in zip(codes, descriptions):
                code_str = str(code).strip().strip('"')
                desc_str = str(desc).strip().strip('"')
                if code_str and code_str != 'IDRSSD' and pd.notna(desc) and not desc_str.isdigit():
                    local_map[code_str] = desc_str
    except Exception:
        pass
    finally:
        file_obj.seek(0)
    return local_map

def clean_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Applies standard cleaning: whitespace stripping, IDRSSD validation, duplication removal.
    Returns a new DataFrame or None if validation fails.
    """
    if df.empty:
        return None

    # Clean column names
    df.columns = [str(col).strip().strip('"') for col in df.columns]

    if 'IDRSSD' not in df.columns:
        return df 

    # Standardize IDRSSD
    df['IDRSSD'] = df['IDRSSD'].astype(str).str.strip()
    
    # Filter invalid rows
    df = df[df['IDRSSD'] != '']
    df = df[df['IDRSSD'] != 'nan']
    
    if df.empty:
        return None

    # Drop rows where all data columns (non-IDRSSD) are NaN
    non_idrssd_cols = df.columns.difference(['IDRSSD'])
    if not non_idrssd_cols.empty:
        df = df.dropna(subset=non_idrssd_cols, how='all')

    if df.empty:
        return None

    # Deduplicate
    if df['IDRSSD'].duplicated().any():
        df = df.drop_duplicates(subset='IDRSSD', keep='first')

    # Set Index
    try:
        return df.set_index('IDRSSD')
    except KeyError:
        return df

def load_dataframe(file_obj, filename: str) -> Tuple[Optional[pd.DataFrame], FileShapeLog]:
    """
    Loads CSV data from a file object. Handles encoding fallbacks.
    Returns Tuple: (Loaded DataFrame or None, ShapeLog).
    """
    read_params = {
        'sep': '\t', 'low_memory': False, 'header': 0, 
        'skiprows': [1], 'encoding': 'utf-8', 'dtype': {'IDRSSD': str}
    }
    
    df = None
    try:
        df = pd.read_csv(file_obj, **read_params)
    except UnicodeDecodeError:
        file_obj.seek(0)
        read_params['encoding'] = 'latin1'
        try:
            df = pd.read_csv(file_obj, **read_params)
        except Exception:
            pass
    except ValueError:
        # Retry with object dtype if specific dtype fails
        file_obj.seek(0)
        read_params.pop('dtype', None)
        try:
            df = pd.read_csv(file_obj, **read_params, dtype=object)
        except Exception:
            pass
    except Exception:
        pass

    rows, cols = (df.shape[0], df.shape[1]) if df is not None else (0, 0)
    return df, FileShapeLog(filename, rows, cols)

def process_single_file_content(file_obj, filename: str) -> ProcessedFileResult:
    """
    Orchestrates parsing, loading, and cleaning for a single file stream.
    """
    metadata = parse_header_metadata(file_obj)
    raw_df, shape_log = load_dataframe(file_obj, filename)
    cleaned_df = clean_dataframe(raw_df.copy()) if raw_df is not None else None
    
    # Metrics calculated on best available data (cleaned or raw)
    target_metric_df = cleaned_df if cleaned_df is not None else raw_df
    metrics = calculate_metrics(target_metric_df, filename)

    match = Config.SPLIT_FILE_PATTERN.match(filename)
    is_split = bool(match)
    base, part, total = match.groups() if match else (None, 0, 0)

    return ProcessedFileResult(
        filename=filename,
        dataframe=cleaned_df,
        metrics=metrics,
        shape_log=shape_log,
        metadata_map=metadata,
        is_split_part=is_split,
        split_base_name=base.strip() if base else None,
        split_part_num=int(part),
        split_total_parts=int(total)
    )

# --- Aggregation Logic ---

def combine_split_groups(results: List[ProcessedFileResult]) -> Dict[str, pd.DataFrame]:
    """
    Identifies split file groups, combines them, and organizes standalone files.
    Returns a Dictionary of {UniqueKey: DataFrame}.
    """
    final_dfs: Dict[str, pd.DataFrame] = {}
    split_groups: Dict[Tuple[str, int], Dict[int, pd.DataFrame]] = {}

    # Segregate
    for res in results:
        if res.dataframe is None or res.dataframe.empty:
            continue
            
        if res.is_split_part:
            key = (res.split_base_name, res.split_total_parts)
            if key not in split_groups:
                split_groups[key] = {}
            split_groups[key][res.split_part_num] = res.dataframe
        else:
            key = Path(res.filename).stem
            final_dfs[_make_unique_key(key, final_dfs)] = res.dataframe

    # Process Split Groups
    for (base_name, total_parts), parts_dict in split_groups.items():
        if len(parts_dict) == total_parts and all(i in parts_dict for i in range(1, total_parts + 1)):
            sorted_dfs = [parts_dict[i] for i in range(1, total_parts + 1)]
            combined = _concat_dfs_horizontally(sorted_dfs)
            key = f"{base_name}_Combined_{total_parts}parts"
            final_dfs[_make_unique_key(key, final_dfs)] = combined
        else:
            # Incomplete group: treat as orphans
            for part_num, df in parts_dict.items():
                key = f"{base_name}_Part_{part_num}_of_{total_parts}"
                final_dfs[_make_unique_key(key, final_dfs)] = df

    return final_dfs

def _concat_dfs_horizontally(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Helper to concat split parts horizontally, handling column overlap."""
    base = dfs[0]
    for i in range(1, len(dfs)):
        next_df = dfs[i]
        common = base.columns.intersection(next_df.columns)
        if not common.empty:
            next_df = next_df.drop(columns=common)
        base = pd.concat([base, next_df], axis=1)
    return base

def _make_unique_key(key: str, existing_keys: Dict) -> str:
    """Ensures dictionary key uniqueness."""
    if key not in existing_keys:
        return key
    counter = 1
    while f"{key}_{counter}" in existing_keys:
        counter += 1
    return f"{key}_{counter}"

def _select_merge_base(dfs: Dict[str, pd.DataFrame]) -> Optional[str]:
    """Determines the best DataFrame to serve as the left-side base for merging."""
    if not dfs:
        return None
    
    keys = list(dfs.keys())
    
    combined = [k for k in keys if "_Combined_" in k]
    if combined:
        return combined[0]
    
    for priority in Config.MERGE_PRIORITY_KEYS:
        for k in keys:
            if priority in k or f"_{priority}" in k:
                return k
                
    return max(keys, key=lambda k: dfs[k].size)

def merge_zip_dataframes(dfs_map: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Merges all DataFrames in the map using Outer Join on IDRSSD index.
    """
    base_key = _select_merge_base(dfs_map)
    if not base_key:
        return None

    merged_df = dfs_map.pop(base_key)
    
    for key, df in dfs_map.items():
        if df.shape[1] > 0: 
            merged_df = pd.merge(
                merged_df, df,
                left_index=True, right_index=True,
                how='outer',
                suffixes=('', f'_{key}')
            )
            
    return merged_df.reset_index()

# --- I/O and Side Effects ---

def write_logs(metrics: List[ProcessingMetrics], 
               metadata: Dict[str, str], 
               shapes: List[FileShapeLog]):
    """Writes the three CSV log files."""
    if metrics:
        pd.DataFrame([asdict(m) for m in metrics]).to_csv("import_debug_log.csv", index=False)
        print("Debug log saved.")
        
    if metadata:
        pd.DataFrame(list(metadata.items()), columns=['Code', 'Metadata']).to_csv("code_metadata_mapping.csv", index=False)
        print("Metadata mapping saved.")

    if shapes:
        pd.DataFrame([asdict(s) for s in shapes]).to_csv("txt_import_shapes.csv", index=False)
        print("Shape log saved.")

def save_parquet(df: pd.DataFrame, original_zip: Path):
    """
    Saves DataFrame to parquet.
    CRITICAL FIX: Converts all data to strings to prevent PyArrow type inference failures 
    caused by FFIEC data mixing integers with text footnotes.
    """
    if df is None or df.empty:
        return

    Config.PARQUET_DIR.mkdir(exist_ok=True)
    output_path = Config.PARQUET_DIR / f"{original_zip.stem}.parquet"
    
    # Force all columns to string to avoid Arrow "Expected bytes, got int" or "Conversion failed" errors.
    # This preserves footnotes like "-597 for 2011 adjustments..." instead of crashing.
    df_str = df.astype(str)

    try:
        df_str.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"Saved: {output_path}")
    except ImportError:
        try:
            df_str.to_parquet(output_path, index=False, engine='fastparquet')
            print(f"Saved (fastparquet): {output_path}")
        except Exception as e:
            print(f"Failed to save {output_path}: {e}")
    except Exception as e:
        print(f"Failed to save {output_path}: {e}")

def process_zip_archive(zip_path: Path) -> Tuple[List[ProcessingMetrics], Dict[str, str], List[FileShapeLog]]:
    """
    Main logic for a single zip file.
    """
    print(f"Processing: {zip_path.name}")
    results: List[ProcessedFileResult] = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            txt_files = [f for f in zf.namelist() if f.lower().endswith('.txt')]
            
            for txt_file in txt_files:
                with zf.open(txt_file) as f:
                    results.append(process_single_file_content(f, Path(txt_file).name))
                    
    except zipfile.BadZipFile:
        print(f"Corrupt zip file: {zip_path}")
        return [], {}, []

    dfs_map = combine_split_groups(results)
    merged_df = merge_zip_dataframes(dfs_map)
    
    if merged_df is not None:
        save_parquet(merged_df, zip_path)

    metrics = [r.metrics for r in results]
    shapes = [r.shape_log for r in results]
    metadata = {k: v for r in results for k, v in r.metadata_map.items()}

    return metrics, metadata, shapes

def main():
    """Entry point."""
    if not Config.ZIP_DIR.exists():
        print(f"Directory '{Config.ZIP_DIR}' not found.")
        return

    zip_files = list(Config.ZIP_DIR.glob('*.zip'))
    
    if Config.DEBUG_MODE:
        zip_files = zip_files[:Config.DEBUG_LIMIT]
        print(f"Debug Mode: Processing first {len(zip_files)} files.")

    all_metrics: List[ProcessingMetrics] = []
    all_metadata: Dict[str, str] = {}
    all_shapes: List[FileShapeLog] = []

    for zip_file in zip_files:
        m, meta, s = process_zip_archive(zip_file)
        all_metrics.extend(m)
        all_metadata.update(meta)
        all_shapes.extend(s)

    write_logs(all_metrics, all_metadata, all_shapes)


if __name__ == '__main__':
    main()