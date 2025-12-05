import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Callable

# Import functions from your script (assuming script is named fdic_etl.py)
# If the script is in the same directory, this import works.
# Run pytest data/fdic/test_data_and_logic.py
from prepare_fdic_refactored import (
    adjust_quarterly_flows, 
    consolidate_rcon_rcfd, 
    normalize_types, 
    FDICTaxonomy, 
    FDICConfig
)

# -----------------------------------------------------------------------------
# PART 1: Data Integrity Tests (Against Official Aggregates)
# -----------------------------------------------------------------------------

# Source: FDIC Quarterly Banking Profile (QBP)
# Units in Call Reports are typically in Thousands ($000s).
# Official QBP numbers are usually in Billions. 
# Conversion factor: 1 Billion = 1,000,000 Thousands.
BILLION_TO_THOUSANDS = 1_000_000

@pytest.fixture(scope="module")
def processed_data() -> pd.DataFrame:
    """
    Loads the processed parquet file once for all integrity tests.
    """
    config = FDICConfig()
    path = config.processed_output_path
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        pytest.skip(f"Processed file not found at {path}. Run ETL first.")

@pytest.fixture
def official_aggregates() -> Dict[str, Dict[str, float]]:
    """
    Returns a dictionary of official aggregate stats for validation.
    Keys are 'YYYY-MM-DD'. Values are in Billions USD (raw QBP data).
    """
    return {
        # Example Data points from FDIC QBP (from official site for exacts)
        '2022-12-31': {
            'total_assets': 23_598.511,  
            'net_income': 68.215        
        },
        '2023-12-31': {
            'total_assets': 23_668.802, 
            'net_income': 38.392         
        }
    }

def validate_metric(
    df: pd.DataFrame, 
    date_str: str, 
    col_name: str, 
    expected_billions: float, 
    tolerance: float = 0.05
) -> None:
    """
    Pure validation function. 
    Asserts that df sum matches expected value within tolerance.
    """
    # Filter by date
    date_mask = df['date'] == pd.to_datetime(date_str)
    if not date_mask.any():
        pytest.skip(f"Date {date_str} not present in dataset.")

    # Calculate Sum in Billions (converting from Thousands)
    actual_total_thousands = df.loc[date_mask, col_name].sum()
    actual_billions = actual_total_thousands / BILLION_TO_THOUSANDS
    
    # Calculate Delta
    diff = abs(actual_billions - expected_billions)
    pct_error = diff / expected_billions

    error_msg = (
        f"Date: {date_str} | Metric: {col_name}\n"
        f"Expected: ${expected_billions}B\n"
        f"Actual:   ${actual_billions:.2f}B\n"
        f"Error:    {pct_error:.2%}"
    )
    
    assert pct_error < tolerance, error_msg

def test_aggregate_assets(processed_data, official_aggregates):
    """Checks Total Assets against QBP reference data."""
    for date_str, metrics in official_aggregates.items():
        if 'total_assets' in metrics:
            validate_metric(
                processed_data, 
                date_str, 
                'total_assets', 
                metrics['total_assets']
            )

def test_aggregate_net_income(processed_data, official_aggregates):
    """Checks Net Income against QBP reference data."""
    for date_str, metrics in official_aggregates.items():
        if 'net_income' in metrics:
            validate_metric(
                processed_data, 
                date_str, 
                'net_income', 
                metrics['net_income'],
                tolerance=0.10  # Income is more volatile than assets
            )

# -----------------------------------------------------------------------------
# PART 2: Unit Tests (Logic Verification)
# -----------------------------------------------------------------------------

def test_adjust_quarterly_flows_logic():
    """
    Verifies that YTD data is correctly converted to quarterly flows.
    Scenario:
    Q1 YTD: 100 -> Q1 Discrete: 100
    Q2 YTD: 250 -> Q2 Discrete: 150 (250 - 100)
    Q3 YTD: 300 -> Q3 Discrete: 50  (300 - 250)
    New Year Q1: 50 -> Q1 Discrete: 50
    """
    # Setup
    cols = ['income']
    data = {
        'IDRSSD': [1, 1, 1, 1],
        'RCON9999': [
            datetime(2023, 3, 31), 
            datetime(2023, 6, 30), 
            datetime(2023, 9, 30),
            datetime(2024, 3, 31) # New year reset
        ],
        'income': [100, 250, 300, 50] # YTD Values
    }
    df_input = pd.DataFrame(data)
    
    # Execute
    result = adjust_quarterly_flows(df_input, cols)
    
    # Assert
    expected_flows = [100.0, 150.0, 50.0, 50.0]
    np.testing.assert_array_equal(result['income'].values, expected_flows)

def test_consolidate_rcon_rcfd_logic():
    """
    Verifies RCON (Domestic) takes precedence, RCFD (Consolidated) fills gaps.
    """
    data = {
        'RCON2170': [100, None, 100],
        'RCFD2170': [500, 200, None]
    }
    df_input = pd.DataFrame(data)
    
    result = consolidate_rcon_rcfd(df_input)
    
    # RCON2170 should remain RCON2170
    # RCFD2170 should be dropped
    assert 'RCFD2170' not in result.columns
    assert 'RCON2170' in result.columns
    
    # Row 0: RCON present (100) -> keep 100
    # Row 1: RCON NaN, RCFD (200) -> fill 200
    # Row 2: RCON present (100), RCFD NaN -> keep 100
    expected = [100.0, 200.0, 100.0]
    np.testing.assert_array_equal(result['RCON2170'].values, expected)

def test_normalize_types_conversion():
    """Verifies string to numeric coercion handles garbage gracefully."""
    taxonomy = FDICTaxonomy()
    data = {
        'IDRSSD': ['1001', '1002'],      # Should remain string
        'RCON9999': ['20230331', '20230630'], # Should become datetime
        'RCON2170': ['5000', 'invalid']  # Should become numeric, invalid -> NaN
    }
    df_input = pd.DataFrame(data)
    
    result = normalize_types(df_input, taxonomy)
    
    assert pd.api.types.is_string_dtype(result['IDRSSD'])
    assert pd.api.types.is_datetime64_any_dtype(result['RCON9999'])
    assert pd.api.types.is_numeric_dtype(result['RCON2170'])
    
    assert pd.isna(result['RCON2170'].iloc[1])  # 'invalid' should be NaN