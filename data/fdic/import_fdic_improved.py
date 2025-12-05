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
    # Changed output directory to 'intermediate' as requested
    INTERMEDIATE_DIR = Path("data/fdic/intermediate") 
    DEBUG_MODE = False
    DEBUG_LIMIT = 5 # Increased limit slightly for context
    SPLIT_FILE_PATTERN = re.compile(r"^(.*?)\s*\((\d+)\s+of\s+(\d+)\)\.txt$", re.IGNORECASE)
    DATE_PATTERNS = ['%Y%m%d', '%m%d%Y']
    
    # Filter patterns mapping to output filenames
    # Note: Whitespace at the end is significant as requested
    TARGET_PATTERNS = {
        "Call Bulk POR ": "bank_information",
        "Call Schedule ENT ": "ENT",
        "Call Schedule RI ": "RI",
        "Call Schedule RC ": "RC"
    }

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
    Returns format 'Qx YYYY' or 'Unknown'.
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

def format_quarter_for_filename(quarter_str: str) -> str:
    """Converts 'Q1 2007' to '2007_Q1' for sorting."""
    if quarter_str == "Unknown":
        return "Unknown_Date"
    try:
        # Expects "Qx YYYY"
        parts = quarter_str.split()
        return f"{parts[1]}_{parts[0]}"
    except IndexError:
        return "Unknown_Date"

def calculate_metrics(df: Optional[pd.DataFrame], filename: str) -> ProcessingMetrics:
    """Calculates data quality metrics for a DataFrame."""
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
    """Reads first two rows of a file object to extract metadata."""
    local_map = {}
    try:
        file_obj.seek(0)
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
    """Applies standard cleaning: whitespace stripping, IDRSSD validation."""
    if df.empty:
        return None

    df.columns = [str(col).strip().strip('"') for col in df.columns]

    if 'IDRSSD' not in df.columns:
        return df 

    df['IDRSSD'] = df['IDRSSD'].astype(str).str.strip()
    df = df[df['IDRSSD'] != '']
    df = df[df['IDRSSD'] != 'nan']
    
    if df.empty:
        return None

    non_idrssd_cols = df.columns.difference(['IDRSSD'])
    if not non_idrssd_cols.empty:
        df = df.dropna(subset=non_idrssd_cols, how='all')

    if df.empty:
        return None

    if df['IDRSSD'].duplicated().any():
        df = df.drop_duplicates(subset='IDRSSD', keep='first')

    try:
        return df.set_index('IDRSSD')
    except KeyError:
        return df

def load_dataframe(file_obj, filename: str) -> Tuple[Optional[pd.DataFrame], FileShapeLog]:
    """Loads CSV data from a file object."""
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
    """Orchestrates parsing for a single file stream."""
    metadata = parse_header_metadata(file_obj)
    raw_df, shape_log = load_dataframe(file_obj, filename)
    cleaned_df = clean_dataframe(raw_df.copy()) if raw_df is not None else None
    
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
    """Identifies split file groups and combines them."""
    final_dfs: Dict[str, pd.DataFrame] = {}
    split_groups: Dict[Tuple[str, int], Dict[int, pd.DataFrame]] = {}

    for res in results:
        if res.dataframe is None or res.dataframe.empty:
            continue
            
        if res.is_split_part:
            key = (res.split_base_name, res.split_total_parts)
            if key not in split_groups:
                split_groups[key] = {}
            split_groups[key][res.split_part_num] = res.dataframe
        else:
            final_dfs[res.filename] = res.dataframe

    for (base_name, total_parts), parts_dict in split_groups.items():
        if len(parts_dict) == total_parts and all(i in parts_dict for i in range(1, total_parts + 1)):
            sorted_dfs = [parts_dict[i] for i in range(1, total_parts + 1)]
            combined = _concat_dfs_horizontally(sorted_dfs)
            # Reconstruct a filename-like key for consistency
            final_dfs[f"{base_name}.txt"] = combined
        else:
            for part_num, df in parts_dict.items():
                final_dfs[f"{base_name} ({part_num} of {total_parts}).txt"] = df

    return final_dfs

def _concat_dfs_horizontally(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    base = dfs[0]
    for i in range(1, len(dfs)):
        next_df = dfs[i]
        common = base.columns.intersection(next_df.columns)
        if not common.empty:
            next_df = next_df.drop(columns=common)
        base = pd.concat([base, next_df], axis=1)
    return base

# --- I/O and Side Effects ---

def write_logs(metrics: List[ProcessingMetrics], metadata: Dict[str, str], shapes: List[FileShapeLog]):
    if metrics:
        pd.DataFrame([asdict(m) for m in metrics]).to_csv("import_debug_log.csv", index=False)
        print("Debug log saved.")
    if metadata:
        pd.DataFrame(list(metadata.items()), columns=['Code', 'Metadata']).to_csv("code_metadata_mapping.csv", index=False)
        print("Metadata mapping saved.")
    if shapes:
        pd.DataFrame([asdict(s) for s in shapes]).to_csv("txt_import_shapes.csv", index=False)
        print("Shape log saved.")

def save_categorized_parquet(dfs_map: Dict[str, pd.DataFrame]):
    """Saves DataFrames to the intermediate folder, categorized by content type."""
    Config.INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename_key, df in dfs_map.items():
        if df is None or df.empty:
            continue
            
        # Determine content type based on filename patterns
        output_type = None
        for pattern, type_name in Config.TARGET_PATTERNS.items():
            if pattern in filename_key:
                output_type = type_name
                break
        
        if not output_type:
            continue
            
        # Determine quarter from filename
        q_raw = extract_reporting_quarter(filename_key, Config.DATE_PATTERNS)
        date_label = format_quarter_for_filename(q_raw)
        
        output_name = f"{date_label}_{output_type}.parquet"
        output_path = Config.INTERMEDIATE_DIR / output_name
        
        # Ensure IDRSSD is a column (reset index) and convert to string for safety
        df_to_save = df.reset_index()
        df_str = df_to_save.astype(str)
        
        try:
            df_str.to_parquet(output_path, index=False, engine='pyarrow')
            print(f"Saved: {output_path}")
        except Exception as e:
            try:
                df_str.to_parquet(output_path, index=False, engine='fastparquet')
                print(f"Saved (fastparquet): {output_path}")
            except Exception as e2:
                print(f"Failed to save {output_path}: {e2}")

def process_zip_archive(zip_path: Path) -> Tuple[List[ProcessingMetrics], Dict[str, str], List[FileShapeLog]]:
    """Main logic for a single zip file with filtering."""
    print(f"Processing: {zip_path.name}")
    results: List[ProcessedFileResult] = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Filter files strictly based on Config patterns
            txt_files = []
            for f in zf.namelist():
                if not f.lower().endswith('.txt'):
                    continue
                # Check if file matches any of the required patterns
                for pattern in Config.TARGET_PATTERNS.keys():
                    if pattern in f:
                        txt_files.append(f)
                        break
            
            for txt_file in txt_files:
                with zf.open(txt_file) as f:
                    results.append(process_single_file_content(f, Path(txt_file).name))
                    
    except zipfile.BadZipFile:
        print(f"Corrupt zip file: {zip_path}")
        return [], {}, []

    # Combine parts if any (though unlikely for these specific files)
    dfs_map = combine_split_groups(results)
    
    # Save individually instead of merging
    save_categorized_parquet(dfs_map)

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