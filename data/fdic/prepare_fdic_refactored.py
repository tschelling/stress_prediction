from heapq import merge
import os
import gc
import glob
import re
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass, field
from functools import reduce
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from rapidfuzz import process, fuzz, utils
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Configuration & Taxonomy
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FDICConfig:
    """Immutable configuration settings for the ETL pipeline."""
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    input_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intermediate')
    failed_bank_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'failed_bank_list.csv')
    adjust_flows: bool = True
    # Indicates
    match_threshold: int = 85
    
    @property
    def raw_output_path(self) -> str:
        name = (
            "data_raw.parquet" 
            if self.adjust_flows 
            else "data_raw_flows_unadjusted.parquet"
        )
        return os.path.join(self.script_dir, name)

    @property
    def processed_output_path(self) -> str:
        name = (
            "data_processed.parquet" 
            if self.adjust_flows 
            else "data_processed_flows_unadjusted.parquet"
        )
        return os.path.join(self.script_dir, name)


@dataclass(frozen=True)
class FDICTaxonomy:
    """Defines the mapping between raw FDIC codes and friendly names."""
    
    essential_raw_codes: Set[str] = field(default_factory=lambda: {'IDRSSD'})
    
    # Mappings
    time_entity: Dict[str, str] = field(default_factory=lambda: {
        'IDRSSD': 'id', 'RSSD9050': 'cert_id', 'RSSD9017': 'bank_name',
        'RSSD9200': 'state',
        # Removed 'RCON9999': 'date' to prevent collision with injected date column
    })
    
    balance_sheet_suffix: Dict[str, str] = field(default_factory=lambda: {
        "2170": "total_assets", "2122": "total_loans_and_leases",
        "2200": "total_deposits", "3545": "trading_assets",
        "3123": "allowance_for_loan_and_lease_losses", "JJ19": "allowance_for_credit_losses"
    })
    
    deposit_suffix: Dict[str, str] = field(default_factory=lambda: {})

    loan_structure: Dict[str, str] = field(default_factory=lambda: {})

    income_statement_suffix: Dict[str, str] = field(default_factory=lambda: {
        "4107": "interest_income", "4073": "interest_expense",
        "4074": "net_interest_income", "4079": "non_interest_income",
        "4093": "non_interest_expense", "4340": "net_income",
        "JJ33": "provisions_for_credit_losses", "4070": "fiduciary_income",
        "4080": "deposit_charge_income", "A220": "trading_revenue",
        "C888": "inv_banking_fee_commission_income", "B497": "other non_interest_income",
        "4135": "salaries_benefits_expense",
    })

    capital_suffix: Dict[str, str] = field(default_factory=lambda: {})
    rwa_suffix: Dict[str, str] = field(default_factory=lambda: {})
    
    asset_quality: Dict[str, str] = field(default_factory=lambda: {
        "RIAD4635": "total_charge_offs", "RIAD4605": "total_recoveries",
        "RCON1407": "npl", "RCFD1407": "npl"
    })

    @property
    def all_mappings(self) -> Dict[str, str]:
        def _expand(prefixes: List[str], suffixes: Dict[str, str]) -> Dict[str, str]:
            return {f"{p}{s}": name for p in prefixes for s, name in suffixes.items()}

        balance = _expand(["RCON", "RCFD"], self.balance_sheet_suffix)
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
        return [k for k in self.all_mappings.keys() if k.startswith("RIAD")]

# -----------------------------------------------------------------------------
# I/O Functions (Strict Date Enforcement)
# -----------------------------------------------------------------------------

def load_and_merge_quarterly_data(config: FDICConfig, desired_codes: Set[str]) -> pd.DataFrame:
    """
    Scans intermediate directory, groups files by Quarter, merges RI/RC/ENT/Info.
    STRICT: Requires ENT file and RCON9999 for date derivation.
    """
    all_files = glob.glob(os.path.join(config.input_dir, "*_*.parquet"))
    
    quarters = set()
    file_map = {}
    
    for f_path in all_files:
        f_name = os.path.basename(f_path)
        if "code_descriptions" in f_name:
            continue
            
        match = re.match(r"(\d{4}_Q\d+)", f_name)
        if match:
            prefix = match.group(1)
            quarters.add(prefix)
            if prefix not in file_map:
                file_map[prefix] = {}
            
            if "bank_information" in f_name:
                file_map[prefix]['info'] = f_path
            elif "_RI" in f_name:
                file_map[prefix]['RI'] = f_path
            elif "_RC" in f_name:
                file_map[prefix]['RC'] = f_path
            elif "_ENT" in f_name:
                file_map[prefix]['ENT'] = f_path

    combined_dfs = []
    print(f"Found {len(quarters)} reporting periods to process.")
    
    for q_prefix in sorted(list(quarters)):
        print(f"Processing {q_prefix}...")
        q_files = file_map[q_prefix]
        
        if 'ENT' not in q_files:
            raise FileNotFoundError(f"Missing 'Call Schedule ENT' file for quarter {q_prefix}. Cannot derive RCON9999 (Date). Aborting.")

        dfs_to_merge = []
        
        def load_component(path, force_cols=None):
            try:
                df_comp = pd.read_parquet(path)
                # Keep IDRSSD and desired codes
                keep_cols = [c for c in df_comp.columns if c == 'IDRSSD' or c in desired_codes]
                
                # If force_cols needed (like RCON9999), ensure they are added
                if force_cols:
                    extra = [c for c in force_cols if c in df_comp.columns and c not in keep_cols]
                    keep_cols.extend(extra)
                
                # Deduplicate columns (just in case input file is dirty)
                keep_cols = list(dict.fromkeys(keep_cols))
                return df_comp[keep_cols]
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return None

        if 'info' in q_files:
            dfs_to_merge.append(load_component(q_files['info']))
        if 'RI' in q_files:
            dfs_to_merge.append(load_component(q_files['RI']))
        if 'RC' in q_files:
            dfs_to_merge.append(load_component(q_files['RC']))
        
        dfs_to_merge.append(load_component(q_files['ENT'], force_cols=['RCON9999']))
            
        dfs_to_merge = [d for d in dfs_to_merge if d is not None and not d.empty]
        
        if not dfs_to_merge:
            continue
            
        # Merge on IDRSSD
        q_df = reduce(lambda left, right: pd.merge(left, right, on='IDRSSD', how='outer'), dfs_to_merge)
        
        if 'RCON9999' not in q_df.columns:
            raise KeyError(f"Column 'RCON9999' not found in merged data for {q_prefix} despite ENT file presence.")
        
        # Convert RCON9999 to datetime and DROP the code to prevent later collisions
        q_df['date'] = pd.to_datetime(q_df['RCON9999'], format='%Y%m%d', errors='coerce')
        q_df.drop(columns=['RCON9999'], inplace=True, errors='ignore')
        
        combined_dfs.append(q_df)
        
    if not combined_dfs:
        return pd.DataFrame()
    
    # Concatenate and immediately deduplicate columns to ensure clean state
    full_df = pd.concat(combined_dfs, ignore_index=True)
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    return full_df

def change_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    print("Changing data types...")
    df_out = df.copy()
    df_out['cert_id'] = pd.to_numeric(df_out['cert_id'], errors='coerce').astype('Int64')
    return df_out

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        print("DataFrame empty. Skipping save.")
        return
    print(f"Saving {df.shape} to {path}...")
    try:
        if 'date' in df.columns:
            df['date'] = df['date'].astype('datetime64[ns]')
        
        # Final check for duplicate columns before saving
        if df.columns.duplicated().any():
            print(f"Warning: Duplicate columns detected before save: {df.columns[df.columns.duplicated()].tolist()}")
            df = df.loc[:, ~df.columns.duplicated()]

        df.to_parquet(path, index=False, engine='pyarrow', compression='gzip')
        print("Save successful.")
    except Exception as e:
        print(f"Save failed: {e}")

# -----------------------------------------------------------------------------
# Transformation Functions
# -----------------------------------------------------------------------------

def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    print("Normalizing data types...")
    df_out = df.copy()
    
    id_cols = ['IDRSSD', 'RSSD9017', 'RSSD9050']
    for col in id_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str)

    exclude_cols = set(id_cols + ['date'])
    numeric_cols = [c for c in df_out.columns if c not in exclude_cols]
    
    for col in numeric_cols:
        df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
        
    return df_out


def adjust_quarterly_flows(df: pd.DataFrame, flow_codes: List[str]) -> pd.DataFrame:
    if df.empty or 'date' not in df.columns or 'IDRSSD' not in df.columns:
        return df

    target_cols = [c for c in flow_codes if c in df.columns]
    if not target_cols:
        return df

    print(f"Adjusting flows for {len(target_cols)} columns...")
    
    df_sorted = df.sort_values(by=['IDRSSD', 'date']).copy()
    df_sorted['__year'] = df_sorted['date'].dt.year

    diffs = df_sorted.groupby(['IDRSSD', '__year'])[target_cols].diff()
    
    for col in target_cols:
        df_sorted[col] = diffs[col].fillna(df_sorted[col])
        
    return df_sorted.drop(columns=['__year'])


def consolidate_rcon_rcfd(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    rcon_cols = [c for c in df_out.columns if c.startswith("RCON")]
    
    for rcon_col in rcon_cols:
        suffix = rcon_col[4:]
        rcfd_col = f"RCFD{suffix}"
        
        if rcfd_col in df_out.columns:
            df_out[rcon_col] = df_out[rcon_col].fillna(df_out[rcfd_col])
            df_out.drop(columns=[rcfd_col], inplace=True)
            
    return df_out


def apply_renaming(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    actual_mapping = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=actual_mapping)


def filter_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'id' not in df.columns:
        return df
        
    df_clean = df[pd.to_numeric(df['id'], errors='coerce').notna()].copy()
    return df_clean.drop_duplicates(subset=['id', 'date'], keep='first')

def add_information(df:pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    if 'date' in df_out.columns:
        df_out['quarter'] = df_out['date'].dt.to_period('Q')
    return df_out

def enrich_with_failures(df: pd.DataFrame, config: FDICConfig) -> pd.DataFrame:
    
    # Read failed bank list an prepare for merging
    if not os.path.exists(config.failed_bank_file):
        print("Failed bank list not found. Skipping enrichment.")
        return df

    fb = pd.read_csv(config.failed_bank_file, parse_dates=['FAILDATE'])
    fb = fb.rename(columns={
        'FAILDATE': 'fail_date', 'CERT': 'cert_id', 'NAME': 'failed_bank_name', 
        "COST": 'failure_cost', "RESTYPE": "fail_type", 
        "QBFASSET": "failure_total_assets", "QBFDEP": "failure_total_deposits"
    })
    fb['quarter'] = fb['fail_date'].dt.to_period('Q') - 1
    # Set quarter to 1 quarter before failure to align with last reporting quarter
    fb = fb[['quarter', 'cert_id', 'failed_bank_name', 'failure_cost', 'failure_total_assets', 'failure_total_deposits', 'fail_type']]
    fb[['fails_next_quarter']] = 1
    
    df['cert_id'] = pd.to_numeric(df['cert_id'], errors='coerce').astype('Int64')
    
    fb_unique_banks = fb['cert_id'].dropna().unique()
    df_unique_banks = df['cert_id'].dropna().unique()

    merged_banks = pd.merge(
        pd.DataFrame({'cert_id': fb_unique_banks}), 
        pd.DataFrame({'cert_id': df_unique_banks}), 
        on='cert_id', how='right', indicator=True
    )

    df_final = pd.merge(
        df, fb, 
        on=['quarter', 'cert_id'],  
        how='left'
    )
    
    # Print tabulated merged_banks info
    print(merged_banks['_merge'].value_counts())

    return df_final


def calculate_custom_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    cols = ['total_charge_offs', 'total_recoveries']
    if set(cols).issubset(df_out.columns):
        tco = df_out['total_charge_offs'].fillna(0)
        rec = df_out['total_recoveries'].fillna(0)
        df_out['net_charge_offs'] = tco - rec
    return df_out

# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def create_charts(df: pd.DataFrame, config: FDICConfig) -> None:
    if df.empty:
        return

    print("Generating charts...")
    output_dir = config.script_dir
    sns.set_theme(style="whitegrid")

    if 'date' in df.columns and 'net_income' in df.columns:
        plt.figure(figsize=(14, 7))
        df_income = df.groupby('date')['net_income'].sum().reset_index()
        df_income['net_income_billions'] = df_income['net_income'] / 1_000_000
        
        ax = sns.barplot(x='date', y='net_income_billions', data=df_income, color='navy')
        plt.title('Quarterly Net Income (All FDIC-Insured Institutions)')
        plt.xlabel('Quarter')
        plt.ylabel('Net Income ($ Billions)')
        
        dates = df_income['date'].dt.strftime('%Y-%m')
        tick_locs = range(0, len(dates), 4)
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(dates.iloc[tick_locs], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quarterly_net_income.png'))
        plt.close()

    if 'fail_type' in df.columns:
        df_failures = df[df['fail_type'].notna()]
        if not df_failures.empty:
            plt.figure(figsize=(10, 6))
            sns.countplot(x='fail_type', data=df_failures, order=df_failures['fail_type'].value_counts().index)
            plt.title('Bank Failures by Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'bank_failures_by_type.png'))
            plt.close()

# -----------------------------------------------------------------------------
# Pipeline Execution
# -----------------------------------------------------------------------------

def main():
    config = FDICConfig(adjust_flows=True)
    taxonomy = FDICTaxonomy()
    
    print(f"Starting ETL. Input: {config.input_dir}")

    desired_codes = set(taxonomy.all_mappings.keys())
    desired_codes.add('RCON9999')

    raw_df = load_and_merge_quarterly_data(config, desired_codes)
    
    if raw_df.empty:
        print("No data loaded.")
        return
        
    raw_df = normalize_types(raw_df)
    
    if config.adjust_flows:
        raw_df = adjust_quarterly_flows(raw_df, taxonomy.flow_codes)
        
    save_dataframe(raw_df, config.raw_output_path)

    processed_df = (
        raw_df
        .pipe(add_information)
        .pipe(consolidate_rcon_rcfd)
        .pipe(apply_renaming, mapping=taxonomy.all_mappings)
        .pipe(filter_and_deduplicate)
        .pipe(enrich_with_failures, config=config)
        .pipe(calculate_custom_metrics)
    )
    
    if 'cert_id' in processed_df.columns:
        processed_df['cert_id'] = processed_df['cert_id'].astype(str)

    # Set index to quarter and cert_id
    processed_df.set_index(['cert_id', 'quarter'], inplace=True)

    save_dataframe(processed_df, config.processed_output_path)
    print(f'Processed data saved to {config.processed_output_path}.')
    create_charts(processed_df, config)
    print("Processing complete.")

if __name__ == "__main__":
    main()