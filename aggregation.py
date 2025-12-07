
import pandas as pd
from typing import List, Tuple
from data.fdic.prepare_fdic_refactored import FDICTaxonomy

def get_stock_and_flow_columns(df: pd.DataFrame, taxonomy: FDICTaxonomy) -> Tuple[List[str], List[str]]:
    """
    Separates dataframe columns into stock and flow columns based on taxonomy.

    Args:
        df: The input DataFrame.
        taxonomy: The FDICTaxonomy object.

    Returns:
        A tuple containing two lists: stock columns and flow columns.
    """
    
    # Create a reverse mapping from new column names to old FDIC codes
    reverse_mapping = {v: k for k, v in taxonomy.all_mappings.items()}
    
    flow_cols = [
        col for col in df.columns 
        if reverse_mapping.get(col) in taxonomy.flow_codes
    ]
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    stock_cols = [
        col for col in numeric_cols 
        if col not in flow_cols
    ]
    
    return stock_cols, flow_cols

def aggregate_yearly(df: pd.DataFrame, taxonomy: FDICTaxonomy) -> pd.DataFrame:
    """
    Aggregates quarterly data to yearly data.

    - Stocks are represented by their end-of-year value.
    - Flows are summed over the four quarters of the year.

    Args:
        df: The input DataFrame with a MultiIndex ('cert_id', 'quarter').
        taxonomy: The FDICTaxonomy object.

    Returns:
        A DataFrame with yearly aggregated data.
    """
    df_reset = df.reset_index()
    df_reset['year'] = df_reset['quarter'].dt.year
    
    stock_cols, flow_cols = get_stock_and_flow_columns(df, taxonomy)
    
    # For stocks, take the last value of the year
    df_stocks = df_reset.sort_values('quarter').groupby(['cert_id', 'year'])[stock_cols].last()
    
    # For flows, sum the values over the year
    df_flows = df_reset.groupby(['cert_id', 'year'])[flow_cols].sum()
    
    return pd.concat([df_stocks, df_flows], axis=1)

def aggregate_rolling_4q(df: pd.DataFrame, taxonomy: FDICTaxonomy) -> pd.DataFrame:
    """
    Aggregates flows to a rolling 4-quarter sum, keeping quarterly frequency.

    - Stocks are kept at their quarterly values.
    - Flows are summed over the preceding 4 quarters.

    Args:
        df: The input DataFrame with a MultiIndex ('cert_id', 'quarter').
        taxonomy: The FDICTaxonomy object.

    Returns:
        A DataFrame with quarterly data and 4-quarter rolling flows.
    """
    stock_cols, flow_cols = get_stock_and_flow_columns(df, taxonomy)
    
    # Flows are summed over a rolling window of 4 quarters
    df_flows_rolling = df.groupby('cert_id')[flow_cols].rolling(window=4, min_periods=4).sum()
    df_flows_rolling = df_flows_rolling.reset_index(level=0, drop=True)
    
    # Stocks are kept as is
    df_stocks = df[stock_cols]
    
    return pd.concat([df_stocks, df_flows_rolling], axis=1).dropna()
