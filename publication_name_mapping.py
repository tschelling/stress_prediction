PUBLICATION_NAMES = {
    # Target Variables
    'interest_income_to_assets': 'Interest Income / Assets',
    'interest_expense_to_assets': 'Interest Expense / Assets',
    'non_interest_income_to_assets': 'Non-Interest Income / Assets',
    'non_interest_expense_to_assets': 'Non-Interest Expense / Assets',
    'net_charge_offs_to_loans_and_leases': 'Net Charge-Offs / Loans',

    # Bank-Specific Feature Variables
    'deposit_ratio': 'Deposit Ratio',
    'loan_to_asset_ratio': 'Loan-to-Asset Ratio',
    'dep_demand_to_assets': 'Demand Deposits / Assets',
    'log_total_assets': 'Log(Total Assets)',
    'is_structural_break': 'Structural Break Dummy',

    # Macroeconomic Feature Variables
    'cpi_qoq': 'CPI (QoQ %)',
    'gdp_qoq': 'GDP (QoQ %)',
    'unemployment_diff': 'Unemployment Rate (Diff)',
    'household_delinq_diff': 'Household Delinquency Rate (Diff)',
    'tbill_3m_diff': '3M T-Bill Rate (Diff)',
    'tbill_10y_diff': '10Y T-Bill Rate (Diff)',
    'sp500_qoq': 'S&P 500 (QoQ %)',
    'corp_bond_spread_diff': 'Corp. Bond Spread (Diff)',
    'vix_qoq': 'VIX (QoQ %)',

    # Maturity Structure Variables (Deposits)
    'dep_small_3m_less_to_assets': 'Small Deposits (<3m) / Assets',
    'dep_small_3m_1y_to_assets': 'Small Deposits (3m-1y) / Assets',
    'dep_small_1y_3y_to_assets': 'Small Deposits (1y-3y) / Assets',
    'dep_small_3y_more_to_assets': 'Small Deposits (>3y) / Assets',
    'dep_large_3m_less_to_assets': 'Large Deposits (<3m) / Assets',
    'dep_large_3m_1y_to_assets': 'Large Deposits (3m-1y) / Assets',
    'dep_large_1y_3y_to_assets': 'Large Deposits (1y-3y) / Assets',
    'dep_large_3y_more_to_assets': 'Large Deposits (>3y) / Assets',

    # Maturity Structure Variables (Loans)
    'closed_end_first_liens_1_4_res_prop_3m_less_to_assets': 'Residential Loans (<3m) / Assets',
    'closed_end_first_liens_1_4_res_prop_3m_1y_to_assets': 'Residential Loans (3m-1y) / Assets',
    'closed_end_first_liens_1_4_res_prop_1y_3y_to_assets': 'Residential Loans (1y-3y) / Assets',
    'closed_end_first_liens_1_4_res_prop_3y_5y_to_assets': 'Residential Loans (3y-5y) / Assets',
    'closed_end_first_liens_1_4_res_prop_5y_15y_to_assets': 'Residential Loans (5y-15y) / Assets',
    'closed_end_first_liens_1_4_res_prop_15y_more_to_assets': 'Residential Loans (>15y) / Assets',
    'all_other_loans_3m_less_to_assets': 'Other Loans (<3m) / Assets',
    'all_other_loans_3m_1y_to_assets': 'Other Loans (3m-1y) / Assets',
    'all_other_loans_1y_3y_to_assets': 'Other Loans (1y-3y) / Assets',
    'all_other_loans_3y_5y_to_assets': 'Other Loans (3y-5y) / Assets',
    'all_other_loans_5y_15y_to_assets': 'Other Loans (5y-15y) / Assets',
    'all_other_loans_15y_more_to_assets': 'Other Loans (>15y) / Assets',

    # Other calculated ratios
    'loan_to_deposit_ratio': 'Loan-to-Deposit Ratio',
    'equity_to_asset_ratio': 'Equity-to-Asset Ratio',
    'net_interest_margin': 'Net Interest Margin',
    'roe': 'Return on Equity (ROE)',
    'roa': 'Return on Assets (ROA)',
}