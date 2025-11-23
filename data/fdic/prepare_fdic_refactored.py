import os
import gc
import glob
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass, field
from functools import reduce
from unicodedata import numeric
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from rapidfuzz import process, fuzz, utils
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# -----------------------------------------------------------------------------
# Configuration & Taxonomy
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FDICConfig:
    """Immutable configuration settings for the ETL pipeline."""
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    input_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parquet')
    failed_bank_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'failed_bank_list.csv')
    adjust_flows: bool = True  # Enabled to support Net Income calculations
    match_threshold: int = 85
    input_pattern: str = '*.parquet'
    
    @property
    def raw_output_path(self) -> str:
        name = (
            "fdic_data_extracted_raw_codes.parquet" 
            if self.adjust_flows 
            else "fdic_data_extracted_raw_codes_flows_unadjusted.parquet"
        )
        return os.path.join(self.script_dir, name)

    @property
    def processed_output_path(self) -> str:
        name = (
            "fdic_data_extracted_processed.parquet" 
            if self.adjust_flows 
            else "fdic_data_processed_flows_unadjusted.parquet"
        )
        return os.path.join(self.script_dir, name)


@dataclass(frozen=True)
class FDICTaxonomy:
    """Defines the mapping between raw FDIC codes and friendly names."""
    
    # Essential codes required for a file to be processed
    essential_raw_codes: Set[str] = field(default_factory=lambda: {'IDRSSD', 'RCON9999'})
    
    # Mappings
    time_entity: Dict[str, str] = field(default_factory=lambda: {
        'IDRSSD': 'id', 'RSSD9050': 'cert_id', 'RSSD9017': 'bank_name',
        'RCON9999': 'date', 'RSSD9200': 'state'
    })
    
    balance_sheet_suffix: Dict[str, str] = field(default_factory=lambda: {
        "2170": "total_assets", "2122": "total_loans_and_leases",
        "2200": "total_deposits", "3545": "trading_assets",
        "3123": "allowance_for_loan_and_lease_losses", "JJ19": "allowance_for_credit_losses"
    })
    
    deposit_suffix: Dict[str, str] = field(default_factory=lambda: {
        "HK07": "dep_small_3m_less", "HK08": "dep_small_3m_1y",
        "HK09": "dep_small_1y_3y", "HK10": "dep_small_3y_more",
        "HK12": "dep_large_3m_less", "HK13": "dep_large_3m_1y",
        "HK14": "dep_large_1y_3y", "HK15": "dep_large_3y_more",
        "2210": "dep_demand", "2215": "dep_transaction_accounts",
        "2385": "dep_non_transaction_accounts",
    })

    loan_structure: Dict[str, str] = field(default_factory=lambda: {
        "RCONA564": "closed_end_first_liens_1_4_res_prop_3m_less",
        "RCONA565": "closed_end_first_liens_1_4_res_prop_3m_1y",
        "RCONA566": "closed_end_first_liens_1_4_res_prop_1y_3y",
        "RCONA567": "closed_end_first_liens_1_4_res_prop_3y_5y",
        "RCONA568": "closed_end_first_liens_1_4_res_prop_5y_15y",
        "RCONA569": "closed_end_first_liens_1_4_res_prop_15y_more",
        "RCFDA570": "all_other_loans_3m_less",
        "RCFDA571": "all_other_loans_3m_1y",
        "RCFDA572": "all_other_loans_1y_3y",
        "RCFDA573": "all_other_loans_3y_5y",
        "RCFDA574": "all_other_loans_5y_15y",
        "RCFDA575": "all_other_loans_15y_more",
    })

    income_statement_suffix: Dict[str, str] = field(default_factory=lambda: {
        "4107": "interest_income", "4073": "interest_expense",
        "4074": "net_interest_income", "4079": "non_interest_income",
        "4093": "non_interest_expense", "4340": "net_income",
        "JJ33": "provisions_for_credit_losses", "4070": "fiduciary_income",
        "4080": "deposit_charge_income", "A220": "trading_revenue",
        "C888": "inv_banking_fee_commission_income", "B497": "other non_interest_income",
        "4135": "salaries_benefits_expense",
    })

    capital_suffix: Dict[str, str] = field(default_factory=lambda: {"G105": "total_equity"})
    rwa_suffix: Dict[str, str] = field(default_factory=lambda: {'G641': 'total_rwa'})
    
    asset_quality: Dict[str, str] = field(default_factory=lambda: {
        "RIAD4635": "total_charge_offs", "RIAD4605": "total_recoveries",
        "RCON1407": "npl", "RCFD1407": "npl"
    })

    @property
    def all_mappings(self) -> Dict[str, str]:
        """Generates the complete dictionary of raw_code -> friendly_name."""
        
        def _expand(prefixes: List[str], suffixes: Dict[str, str]) -> Dict[str, str]:
            return {f"{p}{s}": name for p in prefixes for s, name in suffixes.items()}

        # Construct full mappings
        balance = _expand(["RCON", "RCFD"], self.balance_sheet_suffix)
        # Remove excluded code explicitly mentioned in original script
        balance.pop("RCFD2200", None) 
        
        deposits = _expand(["RCON"], self.deposit_suffix)
        income = _expand(["RIAD"], self.income_statement_suffix)
        capital = _expand(["RCON", "RCFD"], self.capital_suffix)
        rwa = _expand(["RCON", "RCFD"], self.rwa_suffix)

        full_map = {
            **self.time_entity,
            **balance,
            **deposits,
            **self.loan_structure,
            **income,
            **capital,
            **rwa,
            **self.asset_quality
        }
        return full_map

    @property
    def flow_codes(self) -> List[str]:
        """Returns list of codes that are flow variables (Income Statement)."""
        return [k for k in self.all_mappings.keys() if k.startswith("RIAD")]

# -----------------------------------------------------------------------------
# I/O Functions
# -----------------------------------------------------------------------------

def get_valid_files(config: FDICConfig, taxonomy: FDICTaxonomy) -> List[str]:
    """
    Scans directory and returns a list of file paths that contain 
    essential columns defined in the taxonomy.
    """
    search_path = os.path.join(config.input_dir, config.input_pattern)
    all_files = glob.glob(search_path)
    
    # Filter out output files if they exist in the input directory
    ignored_paths = {config.raw_output_path, config.processed_output_path}
    candidates = [f for f in all_files if f not in ignored_paths]
    
    valid_files = []
    for f in candidates:
        try:
            schema = pq.read_schema(f)
            cols = set(schema.names)
            if taxonomy.essential_raw_codes.issubset(cols):
                # Ensure at least one desired code exists beyond essential ones
                if not set(taxonomy.all_mappings.keys()).isdisjoint(cols):
                    valid_files.append(f)
        except Exception as e:
            print(f"Warning: Could not read schema for {f}: {e}")
            
    return sorted(valid_files)


def load_single_file(file_path: str, desired_codes: Set[str]) -> pd.DataFrame:
    """Loads specific columns from a single parquet file."""
    try:
        schema = pq.read_schema(file_path)
        available_cols = set(schema.names)
        # Intersection of desired codes and what is actually in the file
        load_cols = list(desired_codes.intersection(available_cols))
        
        if not load_cols:
            return pd.DataFrame()
            
        df = pd.read_parquet(file_path, columns=load_cols, engine='fastparquet')
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Saves DataFrame to Parquet with compression."""
    if df.empty:
        print("DataFrame empty. Skipping save.")
        return
    print(f"Saving {df.shape} to {path}...")
    try:
        df.to_parquet(path, index=False, engine='pyarrow', compression='gzip')
        print("Save successful.")
    except Exception as e:
        print(f"Save failed: {e}")

# -----------------------------------------------------------------------------
# Transformation Functions (Pure)
# -----------------------------------------------------------------------------

def normalize_types(df: pd.DataFrame, taxonomy: FDICTaxonomy) -> pd.DataFrame:
    """
    Converts columns to appropriate types (Datetime, String, Numeric).
    Returns a new DataFrame.
    """
    df_out = df.copy()
    
    # 1. Convert Dates
    date_col = 'RCON9999'
    if date_col in df_out.columns:
        df_out[date_col] = pd.to_datetime(df_out[date_col], errors='coerce', format='%Y%m%d')

    # 2. Convert Identifiers
    id_cols = ['IDRSSD', 'RSSD9017', 'RSSD9050']
    for col in id_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str)

    # 3. Convert Numeric
    # Exclude date and identifiers from numeric conversion
    exclude_numeric = set(id_cols + [date_col])
    numeric_cols = [c for c in df_out.columns if c not in exclude_numeric]
    
    for col in numeric_cols:
        # Force numeric, coerce errors to NaN
        df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
        
    return df_out


def adjust_quarterly_flows(df: pd.DataFrame, flow_codes: List[str]) -> pd.DataFrame:
    """
    Converts YTD flow data to quarterly discrete data.
    Logic: $Q_t = YTD_t - YTD_{t-1}$ within the same year.
    """
    if df.empty or 'RCON9999' not in df.columns or 'IDRSSD' not in df.columns:
        return df

    # Filter for flow codes actually present in the DF
    target_cols = [c for c in flow_codes if c in df.columns]
    if not target_cols:
        return df

    print(f"Adjusting flows for {len(target_cols)} columns...")
    
    df_sorted = df.sort_values(by=['IDRSSD', 'RCON9999']).copy()
    df_sorted['__year'] = df_sorted['RCON9999'].dt.year

    # Group by Bank and Year, then take diff. 
    # The first record of a year becomes NaN via diff(), which is correct for YTD logic 
    # (Q1 YTD is the Q1 value, so we fillna with original).
    
    # Calculate diffs
    diffs = df_sorted.groupby(['IDRSSD', '__year'])[target_cols].diff()
    
    # If diff is NaN (start of group), use original value (YTD Q1), else use diff
    for col in target_cols:
        df_sorted[col] = diffs[col].fillna(df_sorted[col])
        
    return df_sorted.drop(columns=['__year'])


def consolidate_rcon_rcfd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges paired RCON (domestic) and RCFD (consolidated) columns.
    Priority: RCON. If RCON is NaN, fill with RCFD. Then drop RCFD.
    """
    df_out = df.copy()
    rcon_cols = [c for c in df_out.columns if c.startswith("RCON")]
    
    for rcon_col in rcon_cols:
        suffix = rcon_col[4:]
        rcfd_col = f"RCFD{suffix}"
        
        if rcfd_col in df_out.columns:
            # Fill NA in RCON with RCFD values
            df_out[rcon_col] = df_out[rcon_col].fillna(df_out[rcfd_col])
            df_out.drop(columns=[rcfd_col], inplace=True)
            
    return df_out


def apply_renaming(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Renames columns based on taxonomy."""
    return df.rename(columns=mapping)


def filter_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Removes rows where 'id' is non-numeric.
    2. Deduplicates based on ID and Date (keeping first).
    """
    if df.empty:
        return df
        
    df_clean = df[pd.to_numeric(df['id'], errors='coerce').notna()].copy()
    
    # Ensure date is period for deduplication context if needed, 
    # but keeping as datetime is safer for standard operations until final step.
    return df_clean.drop_duplicates(subset=['id', 'date'], keep='first')


def enrich_with_failures(df: pd.DataFrame, config: FDICConfig) -> pd.DataFrame:
    """
    Loads failed bank list, performs fuzzy matching, and merges failure data.
    """
    if not os.path.exists(config.failed_bank_file):
        print("Failed bank list not found. Skipping enrichment.")
        return df

    # Load and prep failed banks
    fb = pd.read_csv(config.failed_bank_file, parse_dates=['FAILDATE'])
    fb = fb.rename(columns={
        'FAILDATE': 'date', 
        'CERT': 'cert_id', 
        'NAME': 'bank_name', 
        "COST": 'cost',
        "RESTYPE": "fail_type", 
        "QBFASSET": "total_assets_failure", 
        "QBFDEP": "total_deposits_failure"
    })
    # Convert to quarterly period for matching
    fb['date_period'] = fb['date'].dt.to_period('Q')
    
    # Prepare main DF banks list
    unique_banks = df['bank_name'].dropna().unique()

    # Fuzzy Matching Logic
    def get_match(name):
        match = process.extractOne(
            name, unique_banks, 
            scorer=fuzz.token_sort_ratio, 
            processor=utils.default_process, 
            score_cutoff=config.match_threshold
        )
        return (match[0], match[1]) if match else (None, None)

    print("Performing fuzzy matching on failed banks...")
    matches = fb['bank_name'].apply(get_match)
    fb['matched_bank_name'] = matches.apply(lambda x: x[0])
    
    a = df['cert_id'].drop_duplicates()

    # Filter only matched
    fb_matched = fb.dropna(subset=['matched_bank_name'])
    
    # Merge
    # We match on Period and Name. Ensure main DF has a period column for merge
    df_merged = df.copy()
    df_merged['date_period'] = pd.to_datetime(df_merged['date']).dt.to_period('Q')
    
    merge_cols = ['matched_bank_name', 'date_period', 'fail_type', 'total_assets_failure', 'total_deposits_failure']
    fb_subset = fb_matched[merge_cols]
    
    df_final = pd.merge(
        df_merged, 
        fb_subset, 
        left_on=['bank_name', 'date_period'], 
        right_on=['matched_bank_name', 'date_period'], 
        how='left'
    )
    
    print(f"Matched {len(fb_matched)} out of {len(fb)} failed banks.")
    return df_final.drop(columns=['matched_bank_name', 'date_period'])


def calculate_custom_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates derived metrics like net_charge_offs."""
    df_out = df.copy()
    
    cols = ['total_charge_offs', 'total_recoveries']
    if set(cols).issubset(df_out.columns):
        # fillna(0) is usually safer for arithmetic columns derived from flows
        tco = df_out['total_charge_offs'].fillna(0)
        rec = df_out['total_recoveries'].fillna(0)
        df_out['net_charge_offs'] = tco - rec
        
    return df_out

# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def create_charts(df: pd.DataFrame, config: FDICConfig) -> None:
    """
    Generates and saves charts based on the processed DataFrame.
    Adheres to user requirement for Net Income Bar Plots.
    """
    if df.empty:
        print("DataFrame empty. Skipping chart creation.")
        return

    print("Generating charts...")
    output_dir = config.script_dir
    sns.set_theme(style="whitegrid")

    # 1. Bar Chart: Quarterly Net Income (Aggregated)
    if 'date' in df.columns and 'net_income' in df.columns:
        plt.figure(figsize=(14, 7))
        
        # Aggregate net_income by date
        df_income = df.groupby('date')['net_income'].sum().reset_index()
        
        # Convert thousands to Billions for display (Standard FDIC QBP format)
        df_income['net_income_billions'] = df_income['net_income'] / 1_000_000
        
        # Create Bar Plot
        ax = sns.barplot(x='date', y='net_income_billions', data=df_income, color='navy')
        
        plt.title('Quarterly Net Income (All FDIC-Insured Institutions)')
        plt.xlabel('Quarter')
        plt.ylabel('Net Income ($ Billions)')
        
        # Format X-Axis to show fewer labels
        # Seaborn barplot x-axis is categorical (0, 1, 2...), so we need to sample labels
        # Locate ticks for every 4th quarter (Yearly)
        dates = df_income['date'].dt.strftime('%Y-%m')
        tick_locs = range(0, len(dates), 4)
        tick_labels = dates.iloc[tick_locs]
        
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=45)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'quarterly_net_income.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved chart: {output_path}")

    # 2. Bar Chart: Bank Failures by Type
    if 'fail_type' in df.columns:
        df_failures = df[df['fail_type'].notna()]
        if not df_failures.empty:
            plt.figure(figsize=(10, 6))
            sns.countplot(x='fail_type', data=df_failures, order=df_failures['fail_type'].value_counts().index)
            plt.title('Bank Failures by Type')
            plt.xlabel('Failure Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            output_path = os.path.join(output_dir, 'bank_failures_by_type.png')
            plt.savefig(output_path)
            plt.close()
            print(f"Saved chart: {output_path}")
        else:
            print("No bank failures found in the data. Skipping failure chart.")

# -----------------------------------------------------------------------------
# Pipeline Execution
# -----------------------------------------------------------------------------

def main():
    # 1. Initialize Configuration
    # Adjust Flows ENABLED to allow for correct Net Income calculation
    config = FDICConfig(adjust_flows=True) 
    taxonomy = FDICTaxonomy()
    
    print(f"Starting ETL. Output: {config.processed_output_path}")

    # 2. Identify Files
    valid_files = get_valid_files(config, taxonomy)
    if not valid_files:
        print("No valid input files found.")
        return

    # 3. Load & Concat (Map-Reduce)
    print(f"Loading {len(valid_files)} files...")
    desired_codes = set(taxonomy.all_mappings.keys())
    
    dfs = [load_single_file(f, desired_codes) for f in valid_files]
    
    if not dfs:
        print("No data loaded.")
        return
        
    raw_df = pd.concat(dfs, ignore_index=True, sort=False, join='outer')
    del dfs
    gc.collect()
    
    # 4. Normalization & Flow Adjustment
    raw_df = normalize_types(raw_df, taxonomy)
    
    if config.adjust_flows:
        raw_df = adjust_quarterly_flows(raw_df, taxonomy.flow_codes)
        
    # Save Intermediate Raw Data
    save_dataframe(raw_df, config.raw_output_path)

    # 5. Processing Pipeline
    processed_df = (
        raw_df
        .pipe(consolidate_rcon_rcfd)
        .pipe(apply_renaming, mapping=taxonomy.all_mappings)
        .pipe(filter_and_deduplicate)
        .pipe(enrich_with_failures, config=config)
        .pipe(calculate_custom_metrics)
    )
    
    # 6. Final Cleanup
    if 'cert_id' in processed_df.columns:
        processed_df['cert_id'] = processed_df['cert_id'].astype(str)

    # 7. Save Final
    save_dataframe(processed_df, config.processed_output_path)
    
    # 8. Create Charts
    create_charts(processed_df, config)

    print("Processing complete.")

if __name__ == "__main__":
    main()