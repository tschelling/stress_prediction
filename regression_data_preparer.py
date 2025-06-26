import pandas as pd
import numpy as np
from pyparsing import null_debug_action
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from typing import Any, List, Tuple, Optional, Union, Dict # Ensure Dict is imported

class RegressionDataPreparer:
    
    def __init__(self, fred: pd.DataFrame, fdic:pd.DataFrame, yahoo:pd.DataFrame, 
                 config: dict[str, any]):
        self.fred = fred.copy()
        self.fdic = fdic.copy()
        self.yahoo = yahoo.copy()

        self.df0 = None
        self.config = config.copy()

        self.target_variables: List[str] = list(self.config['TARGET_VARIABLES'].keys())
        # Ensure feature_variables list does not contain the current target_variable
        _configured_features = list(self.config.get('FEATURE_VARIABLES').keys())[:]
        self.feature_variables: List[str] = [fv for fv in _configured_features if fv not in self.target_variables]
        self.bank_variables: List[str] = list(var for var, var_type in self.config['FEATURE_VARIABLES'].items() if var_type == 'bank') + list(self.config['TARGET_VARIABLES'].keys())
        # Ensure total_assets is always considered a bank variable for feature engineering
        if 'total_assets' not in self.bank_variables:
            self.bank_variables.append('total_assets')
        self.macro_variables = list(var for var, var_type in self.config['FEATURE_VARIABLES'].items() if var_type == 'macro')
        
        self.include_time_fe: bool = self.config.get('INCLUDE_TIME_FE', False)
        self.include_bank_fe: bool = self.config.get('INCLUDE_BANK_FE', False)
        self.include_ar_lags: bool = self.config.get('INCLUDE_AUTOREGRESSIVE_LAGS', True)
        self.num_lags_to_include: int = self.config.get('NUMBER_OF_LAGS_TO_INCLUDE', 0)
        self.include_structural_break_dummy: bool = self.config.get('INCLUDE_STRUCTURAL_BREAK_DUMMY', False)

        self.train_test_split_dimension = self.config['TRAIN_TEST_SPLIT_DIMENSION']
        self.test_split_config = self.config['TEST_SPLIT']

        self.do_winsorize: bool = self.config.get('WINSORIZE_DO', False)
        self.vars_to_winsorize: List[str] = self.config.get('WINSORIZE_COLS', [])
        self.winsorize_lower_percentile: float = self.config.get('WINSORIZE_LOWER_PERCENTILE', 0.01)
        self.winsorize_upper_percentile: float = self.config.get('WINSORIZE_UPPER_PERCENTILE', 0.99)
        
        self.correct_structural_breaks_total_assets: bool = self.config.get('CORRECT_STRUCTURAL_BREAKS_TOTAL_ASSETS', False)
        self.structural_break_std_dev_threshold: float = self.config.get('STRUCTURAL_BREAK_STD_DEV_THRESHOLD', 3.0)
        
        self.data_processing_stats: Dict[str, Dict[str, Any]] = {} # To store stats from new method


        # DataFrames at different stages of preparation
        # Main preparation steps
        self.df1: Optional[pd.DataFrame] = None
        self.df2: Optional[pd.DataFrame] = None
        self.df3: Optional[pd.DataFrame] = None
        self.df4: Optional[pd.DataFrame] = None
        self.df5: Optional[pd.DataFrame] = None
        self.df6: Optional[pd.DataFrame] = None
        self.df7: Optional[pd.DataFrame] = None
        self.df8: Optional[pd.DataFrame] = None
        self.final_data = None
  

        self.new_features_added_during_prep: List[str] = [] # To track features like 'does_not_report'
        self.final_feature_list: List[str] = self.feature_variables[:]
        self._prepare_initial_data()

    def _prepare_initial_data(self): # No change here, but it's the entry point
        
        print(f"\n--- Merging data FRED, FDIC, yahoo ------------------------------------------------------------")
        df = self.fred.merge(self.fdic.reset_index(), on='date', how='left').merge(self.yahoo, on='date', how='left')
        df.set_index(['id', 'date'], inplace=True)
        self.df0 = df.copy()
        print(f"df0 shape: {self.df0.shape}")
        
        print(f"\n--- Process missing values in index: delete missing values, deduplicate -----------------------")
        df = df[~df.index.get_level_values('id').isna()]
        print(f"df0 shape after index NaN removal: {self.df0.shape}")

        print(f"\n--- Process missing values: Forward fill intermittent missing values --------------------------")
        numeric_fdic_cols = self.fdic.select_dtypes(include=np.number).columns.tolist()
        numeric_fdic_cols = ['total_assets']
        df, fill_summary_df = self._fill_intermittent_missing_values(
            df,
            columns_to_check=numeric_fdic_cols
        )

        self.df1 = df.copy() # After intermittent missing value filling
        print(f"df1 shape: {self.df1.shape}")

        print("--- Feature engineering: Calculate financial ratios -----------------------------------")
        df = self._calculate_financial_ratios(df)

        df = self._engineer_bank_features(df)
        print(f"df shape after financial ratios and bank features: {df.shape}")

        print(f"\n--- Feature selection --------------------------------------------------------------------")

        df = self._select_features_and_target(df)
        print(f"df shape after feature selection: {df.shape}")

        print(f"\n--- Process missing values: Remove remaining missing values --------------------------")
        df, missing_stats = self._process_missing_values(df)
        self.df2 = df.copy() # After remaining missing value processing
        self.df2 = df.copy()
        self.data_processing_stats = missing_stats # Store stats
        print(f"After missing value processing (no imputation), data shape: {df.shape}")


        initial_feature_list = self.feature_variables[:]


        print(f"\n--- Feature generation: Include newly created features --------------------------")
        for new_feat in self.new_features_added_during_prep:
            if new_feat not in initial_feature_list and new_feat in df.columns:
                initial_feature_list.append(new_feat)
        print(f"df shape after including new prep features: {df.shape}")

        # Include quarter fixed effects
        if self.include_time_fe:
            if not pd.api.types.is_datetime64_any_dtype(df.index.get_level_values('date')):
                 df.index = df.index.set_levels(pd.to_datetime(df.index.get_level_values('date')), level='date')
                 print("Converted 'date' index level to datetime.")
            df['quarter'] = 'quarter_' + df.index.get_level_values('date').quarter.astype(str)
            if 'quarter' not in initial_feature_list: initial_feature_list.append('quarter')
            print("Added time fixed effects: 'quarter'.")
        self.df3 = df.copy() # After time FE
        print(f"df3 shape: {self.df3.shape}")

        # Include bank fixed effects if configured
        if self.include_bank_fe:
            df['bank_id'] = ['bank_id_' + str(bank_id) for bank_id in df.index.get_level_values('id')]
            if 'bank_id' not in initial_feature_list:
                initial_feature_list = ['bank_id'] + initial_feature_list 
            print("Added bank fixed effect: 'bank_id'.")
        self.df4 = df.copy() # After bank FE
        print(f"df4 shape: {self.df4.shape}")
        
        self.final_feature_list = initial_feature_list
        
        print(f"\n--- Outlier removal --------------------------")
        if not df.empty:
            df = self._remove_outliers(df, self.target_variables, threshold=self.config.get('OUTLIER_THRESHOLD_TARGET', 3.0))
        else:
            print("DataFrame is empty after FE processing and NaN drop. Skipping outlier removal.")
        self.df5 = df.copy() # After outlier removal
        print(f"df5 shape: {self.df5.shape}")
        

        print(f"\n--- Restrict sample --------------------------")

        if not df.empty:
             df = self._restrict_sample_logic(df)
        else:
            print("DataFrame is empty before sample restriction. Skipping sample restriction.")
        self.df6 = df.copy() # After sample restriction
        print(f"df6 shape: {self.df6.shape}")

        # Correct structural breaks in total assets if configured
        if self.correct_structural_breaks_total_assets:
            print("Applying structural break correction for 'total_assets'...")
            df = self._correct_structural_breaks_in_total_assets(df)
        self.df7 = df.copy() # After structural break correction
        print(f"df7 shape: {self.df7.shape}")

        print(f"\n--- Winsorize data --------------------------")
        df = self._winsorize_data(df) 
        self.df8 = df.copy() # After winsorization
        print(f"df8 shape: {self.df8.shape}")


        self.final_data = df.copy()
        print(f"Base data prepared. Shape: {self.final_data.shape}")
        print(f"Final features for horizon processing: {self.final_feature_list}")

    def _calculate_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates various financial ratios and adds them as new columns to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with necessary columns
                            (e.g., total_deposits, total_assets, etc.).

        Returns:
            pd.DataFrame: DataFrame with added ratio columns.
        """
        print("Calculating financial ratios...")
        df2 = df.copy()

        # Calculate ratios
        df2['deposit_ratio'] = df2['total_deposits'] / df2['total_assets']
        df2['loan_to_deposit_ratio'] = df2['total_loans_and_leases'] / df2['total_deposits']
        df2['loan_to_asset_ratio'] = df2['total_loans_and_leases'] / df2['total_assets']
        df2['equity_to_asset_ratio'] = df2['total_equity'] / df2['total_assets']
        df2['trading_assets_ratio'] = df2['trading_assets'] / df2['total_assets']
        df2['net_interest_margin'] = (
            df2['interest_income'] - df2['interest_expense']
        ) / df2['total_assets']
        df2['roe'] = df2['net_income'] / df2['total_equity']
        df2['roa'] = df2['net_income'] / df2['total_assets']
        df2[         'net_income_to_assets'] =  df2[         'net_income']  / df2['total_assets']
        df2['net_interest_income_to_assets'] =  df2['net_interest_income']  / df2['total_assets']
        df2[    'interest_income_to_assets'] =  df2[    'interest_income']  / df2['total_assets']
        df2[    'interest_expense_to_assets'] = df2[    'interest_expense'] / df2['total_assets']
        df2['non_interest_income_to_assets'] =  df2['non_interest_income']  / df2['total_assets']
        df2['non_interest_expense_to_assets'] = df2['non_interest_expense'] / df2['total_assets']
        df2['net_charge_offs_to_loans_and_leases'] = df2['net_charge_offs'] / df2['total_loans_and_leases']
        df2['npl_ratio'] = df2['npl'] / df2['total_loans_and_leases']
        df2['charge_off_ratio'] = df2['total_charge_offs'] / df2['total_loans_and_leases']
        df2['allowance_for_loan_and_lease_losses_to_assets'] = (
            df2['allowance_for_loan_and_lease_losses'] / df2['total_assets'])
        df2['allowance_for_credit_losses_to_assets'] = (
            df2['allowance_for_credit_losses'] / df2['total_assets'])
        df2['provisions_for_credit_losses_to_assets'] = (
            df2['provisions_for_credit_losses'] / df2['total_assets'])
        df2['rwa_ratio'] = df2['total_rwa'] / df2['total_assets']

        df2['dep_short_term']              = df2['dep_small_3m_less'] + df2['dep_small_3m_1y'] + df2['dep_large_3m_less'] + df2['dep_large_3m_1y']
        df2['dep_long_term']               = df2['dep_small_1y_3y'] + df2['dep_small_3y_more'] + df2['dep_large_1y_3y'] + df2['dep_large_3y_more']
        df2['dep_small_3m_less_to_assets'] = df2['dep_small_3m_less'] / df2['total_assets']
        df2['dep_small_3m_1y_to_assets']   = df2['dep_small_3m_1y']   / df2['total_assets']
        df2['dep_small_1y_3y_to_assets']   = df2['dep_small_1y_3y']   / df2['total_assets']
        df2['dep_small_3y_more_to_assets'] = df2['dep_small_3y_more'] / df2['total_assets']
        df2['dep_large_3m_less_to_assets'] = df2['dep_large_3m_less'] / df2['total_assets']
        df2['dep_large_3m_1y_to_assets']   = df2['dep_large_3m_1y']   / df2['total_assets']
        df2['dep_large_1y_3y_to_assets']   = df2['dep_large_1y_3y']   / df2['total_assets']
        df2['dep_large_3y_more_to_assets'] = df2['dep_large_3y_more'] / df2['total_assets']

        df2['loans_short_term']               = df2['closed_end_first_liens_1_4_res_prop_3m_less'] + \
                                                df2['closed_end_first_liens_1_4_res_prop_3m_1y'] + \
                                                df2['all_other_loans_3m_less'] + \
                                                df2['all_other_loans_3m_1y'] 
        df2['loans_long_term']               =  df2['closed_end_first_liens_1_4_res_prop_1y_3y'] + \
                                                df2['closed_end_first_liens_1_4_res_prop_3y_5y'] + \
                                                df2['closed_end_first_liens_1_4_res_prop_5y_15y'] + \
                                                df2['closed_end_first_liens_1_4_res_prop_15y_more'] + \
                                                df2['all_other_loans_1y_3y'] + \
                                                df2['all_other_loans_3y_5y'] + \
                                                df2['all_other_loans_5y_15y'] + \
                                                df2['all_other_loans_15y_more']
        df2['loans_short_term_to_assets'] = df2['loans_short_term'] / df2['total_assets']
        df2['dep_short_term_to_assets']   = df2['dep_short_term'] /   df2['total_assets']
        df2['closed_end_first_liens_1_4_res_prop_3m_less_to_assets']  = df2['closed_end_first_liens_1_4_res_prop_3m_less']  / df2['total_assets']
        df2['closed_end_first_liens_1_4_res_prop_3m_1y_to_assets']    = df2['closed_end_first_liens_1_4_res_prop_3m_1y']    / df2['total_assets']
        df2['closed_end_first_liens_1_4_res_prop_1y_3y_to_assets']    = df2['closed_end_first_liens_1_4_res_prop_1y_3y']    / df2['total_assets']
        df2['closed_end_first_liens_1_4_res_prop_3y_5y_to_assets']    = df2['closed_end_first_liens_1_4_res_prop_3y_5y']    / df2['total_assets']
        df2['closed_end_first_liens_1_4_res_prop_5y_15y_to_assets']   = df2['closed_end_first_liens_1_4_res_prop_5y_15y']   / df2['total_assets']
        df2['closed_end_first_liens_1_4_res_prop_15y_more_to_assets'] = df2['closed_end_first_liens_1_4_res_prop_15y_more'] / df2['total_assets']
        df2['all_other_loans_3m_less_to_assets']                      = df2['all_other_loans_3m_less']                      / df2['total_assets']
        df2['all_other_loans_3m_1y_to_assets']                        = df2['all_other_loans_3m_1y']                        / df2['total_assets']
        df2['all_other_loans_1y_3y_to_assets']                        = df2['all_other_loans_1y_3y']                        / df2['total_assets']
        df2['all_other_loans_3y_5y_to_assets']                        = df2['all_other_loans_3y_5y']                        / df2['total_assets']
        df2['all_other_loans_5y_15y_to_assets']                       = df2['all_other_loans_5y_15y']                       / df2['total_assets']
        df2['all_other_loans_15y_more_to_assets']                     = df2['all_other_loans_15y_more']                     / df2['total_assets']
        df2['dep_demand_to_assets'] = df2['dep_demand'] / df2['total_assets']
        df2['dep_transaction_accounts_to_assets'] = df2['dep_transaction_accounts'] / df2['total_assets']




        # Log total assets
        df2['log_total_assets'] = np.log(df2['total_assets'].replace(0, np.nan).fillna(1e-9))

        print("Financial ratios calculated.")
        return df2

    def _engineer_bank_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers new features for bank-level variables, like differences and QoQ changes.
        """
        print("Engineering bank-specific features (diffs and QoQ)...")
        df_out = df.sort_index(level=['id', 'date']).copy()

        bank_vars_to_engineer = [var for var in self.bank_variables if var in df_out.columns]
        
        new_features = []
        for col in bank_vars_to_engineer:
            grouped = df_out.groupby(level='id')[col]
            
            diff_col = f"{col}_diff"
            df_out[diff_col] = grouped.diff()
            new_features.append(diff_col)
            
            qoq_col = f"{col}_qoq"
            df_out[qoq_col] = grouped.pct_change()
            # Replace inf/-inf values resulting from pct_change (division by zero) with NaN
            # These will then be handled by subsequent missing value processing steps.
            df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
            new_features.append(qoq_col)

        self.feature_variables.extend([f for f in new_features if f not in self.feature_variables])
        print(f"Engineered {len(new_features)} new bank features.")
        return df_out

    def _select_features_and_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects only the target variable and base feature variables."""
        print("Selecting features and target variable...")
        cols_to_select: List[str] = self.feature_variables[:] + self.target_variables[:]
        # Ensure 'total_assets' is kept if bank size restriction is active
        if self.config.get('RESTRICT_TO_BANK_SIZE') is not None and 'total_assets' not in cols_to_select:
            cols_to_select.append('total_assets')
        # Ensure all selected columns exist in the DataFrame
        existing_cols = [col for col in cols_to_select if col in df.columns]
        missing_cols = [col for col in cols_to_select if col not in df.columns]
        if missing_cols:
            print(f"Warning: The following configured columns are missing from the input DataFrame and will be ignored: {missing_cols}")

        if not existing_cols:
            print("Error: No configured target or feature variables found in the DataFrame. Returning an empty DataFrame.")
            return pd.DataFrame()

        df_selected = df[existing_cols].copy()
        print(f"Selected columns: {df_selected.columns.tolist()}. Shape after selection: {df_selected.shape}")
        return df_selected

    def _process_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Processes missing values.
        1. Removes rows with NaN in 'id' or 'date' index levels.
        - Deduplicates index.
        - Removes banks with not enough data, based on total assets.
        - Returns a DataFrame and information on the state of the data before and after processing.
        """
        stats_summary: Dict[str, Dict[str, Any]] = {}

        # 0. Initial state before any processing in this function
        stats_summary['initial'] = {
            'rows': len(df),
            'banks': df.index.get_level_values('id').nunique(),
            'min_date': df.index.get_level_values('date').min() if not df.empty else None,
            'max_date': df.index.get_level_values('date').max() if not df.empty else None
        }
        print(f"Initial data shape: {df.shape}. Banks: {stats_summary['initial']['banks']}, Dates: {stats_summary['initial']['min_date']} to {stats_summary['initial']['max_date']}")

        # 1. Remove rows where 'id' or 'date' in the index is NaN/NaT
        rows_before_nan_index_removal = len(df)
        df = df[pd.notna(df.index.get_level_values('id')) & pd.notna(df.index.get_level_values('date'))]
        rows_removed_by_nan_index = rows_before_nan_index_removal - len(df)
        stats_summary['nan_index_removal'] = {
            'rows_removed': rows_removed_by_nan_index,
            'remaining_rows': len(df),
            'banks_remaining': df.index.get_level_values('id').nunique(),
            'min_date': df.index.get_level_values('date').min() if not df.empty else None,
            'max_date': df.index.get_level_values('date').max() if not df.empty else None
        }
        print(f"Removed {rows_removed_by_nan_index} rows with NaN in 'id' or 'date' index levels. Remaining rows: {len(df)}. Banks remaining: {stats_summary['nan_index_removal']['banks_remaining']}, Dates: {stats_summary['nan_index_removal']['min_date']} to {stats_summary['nan_index_removal']['max_date']}")

        # 2. Deduplicate index
        duplicated_rows = df.index.duplicated(keep='first')
        if duplicated_rows.any():
            df = df[~duplicated_rows]
            stats_summary['deduplication'] = {
                'rows_removed': duplicated_rows.sum(),
                'remaining_rows': len(df),
                'banks_remaining': df.index.get_level_values('id').nunique(),
                'min_date': df.index.get_level_values('date').min() if not df.empty else None,
                'max_date': df.index.get_level_values('date').max() if not df.empty else None
            }
        
        self.df1 = df.copy()

        # 3. Remove banks with not enough data, based on total assets
        minimum_observations_per_bank = self.config.get('MIN_OBS_PER_BANK', 1)
        if minimum_observations_per_bank > 1:
            nr_rows_before = len(df)
            nr_banks_before = df.index.get_level_values('id').nunique() 
            min_date = df.index.get_level_values('date').min() 
            max_date = df.index.get_level_values('date').max() 
            # Number of observations per bank in total assets (or log_total_assets or deposit_ratio)
            if 'total_assets' in df.columns:
                bank_obs_counts = df.groupby(level='id')['total_assets'].count()
            elif 'log_total_assets' in df.columns:
                bank_obs_counts = df.groupby(level='id')['log_total_assets'].count()
            else:
                bank_obs_counts = df.groupby(level='id')['deposit_ratio'].count()

            banks_to_keep = bank_obs_counts[bank_obs_counts >= minimum_observations_per_bank].index
            df = df[df.index.get_level_values('id').isin(banks_to_keep)]
            nr_banks_after = df.index.get_level_values('id').nunique()
            stats_summary['bank_removal_not_enough_data'] = {
                'minimum_observations_per_bank': minimum_observations_per_bank,
                'banks_before': nr_banks_before,
                'banks_after': nr_banks_after,
                'rows_before': nr_rows_before,
                'rows_after': len(df)   
            }
            print(f"Removed {nr_banks_before - nr_banks_after} banks with less than {minimum_observations_per_bank} observations. Banks before: {nr_banks_before}, after: {nr_banks_after}.") 
            print(f"Removed {nr_rows_before - len(df)} rows based on minimum observations per bank. Rows before: {nr_rows_before}, after: {len(df)}.")

        return df, stats_summary

    def _fill_intermittent_missing_values(
        self, # Add self as the first parameter
        df_input: pd.DataFrame, 
        columns_to_check: Union[str, List[str]],
        id_col: str = 'id',
        date_col: str = 'date'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identifies and forward-fills intermittent missing values in specified columns of a DataFrame.

        An intermittent missing value is defined as a NaN at time t, 
        with non-NaN values at t-1 and t+1 for the same id.
        These intermittent NaNs are filled with the value from t-1.

        Args:
            df_input (pd.DataFrame): Input DataFrame. Must contain id_col and date_col.
            columns_to_check (Union[str, List[str]]): Column name(s) to check and fill.
            id_col (str): Name of the identifier column (e.g., 'id').
            date_col (str): Name of the date column.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - df_filled (pd.DataFrame): The DataFrame with intermittent missing values filled.
                - fill_summary_df (pd.DataFrame): A DataFrame with 'date' and 'count_filled' 
                                                (total intermittent NaNs filled on that date across all checked columns).
        """
        df = df_input.copy()
        df = self.rectangularize_dataframe(df)  # Ensure DataFrame is rectangular
        if isinstance(columns_to_check, str):
            columns_to_check = [columns_to_check]

        all_fill_details = [] # To store (date) for each filled value to create summary

        for col_name in columns_to_check:
            if col_name not in df.columns:
                print(f"Intermittent missing value filling warning: Column '{col_name}' not found in DataFrame. Skipping.")
                continue

            # Sort by index levels if id_col and date_col are part of the index
            df = df.sort_index(level=[id_col, date_col])
            prev_val_col_temp = f'_{col_name}_prev_temp_fill'
            next_val_col_temp = f'_{col_name}_next_temp_fill'
            df[prev_val_col_temp] = df.groupby(level=id_col)[col_name].shift(1)
            df[next_val_col_temp] = df.groupby(level=id_col)[col_name].shift(-1)
            df_check = df[['total_assets', prev_val_col_temp, next_val_col_temp]].copy()
            intermittent_missing_mask = (df[col_name].isna() & df[prev_val_col_temp].notna() & df[next_val_col_temp].notna())
            if intermittent_missing_mask.sum() > 0:
                # Get date values from the index level
                dates_of_fills_for_col = df.loc[intermittent_missing_mask].index.get_level_values(date_col)
                for date_val in dates_of_fills_for_col: all_fill_details.append({'date': date_val})
                df.loc[intermittent_missing_mask, col_name] = df.loc[intermittent_missing_mask, prev_val_col_temp]
            df.drop(columns=[prev_val_col_temp, next_val_col_temp], inplace=True)

        fill_summary_df = pd.DataFrame(columns=[date_col, 'count_filled'])
        if all_fill_details:
            fill_summary_df = pd.DataFrame(all_fill_details).groupby(date_col).size().reset_index(name='count_filled').sort_values(by=date_col)
        # Ensure the output summary DataFrame has a 'date' column as per docstring

        # Revert rectangularization, delete all rows with NaN in total_assets
        df = df[df['total_assets'].notna()]

        return df, fill_summary_df.rename(columns={date_col: 'date'}) if date_col != 'date' and not fill_summary_df.empty else fill_summary_df

    def _winsorize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorizes specified columns of a DataFrame based on configuration."""
        df_out = df.copy()

        if not self.do_winsorize:
            print("Winsorization skipped as per WINSORIZE_DO flag in config.")
            return df_out

        if not self.vars_to_winsorize:
            print("Winsorization skipped: WINSORIZE_COLS list is empty in config.")
            return df_out

        print(f"Winsorizing columns at [{self.winsorize_lower_percentile}, {self.winsorize_upper_percentile}] percentiles...")
        for col in self.vars_to_winsorize:
            if col in df_out.columns:
                if pd.api.types.is_numeric_dtype(df_out[col]):
                    print(f" - Winsorizing {col}")
                    lower_bound = df_out[col].quantile(self.winsorize_lower_percentile)
                    upper_bound = df_out[col].quantile(self.winsorize_upper_percentile)
                    df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)
                else:
                    print(f" - Skipping non-numeric column {col} for winsorization.")
            else:
                print(f" - Skipping winsorization for column {col} (not found in DataFrame).")
        print("Winsorization complete.")
        return df_out

    def _remove_outliers(self, df: pd.DataFrame, target_cols: Union[str, List[str]], threshold: float) -> pd.DataFrame:
        df_filtered = df.copy()
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        print(f"Removing outliers from target variable(s) using z-score threshold {threshold}...")
        
        for target_col in target_cols:
            if target_col not in df_filtered.columns:
                print(f"Warning: Target column '{target_col}' not found in DataFrame. Skipping outlier removal for this column.")
                continue

            if df_filtered.empty:
                print(f"DataFrame is empty before processing '{target_col}'. Skipping.")
                continue
            
            print(f"Processing outliers for '{target_col}'...")
            nr_rows_before = df_filtered.shape[0]
            
            # Ensure column is numeric
            if not pd.api.types.is_numeric_dtype(df_filtered[target_col]):
                print(f"Warning: Column '{target_col}' is not numeric. Skipping outlier removal for this column.")
                continue

            # Calculate z-scores, handle potential division by zero if std is 0
            std_dev = df_filtered[target_col].std()
            if std_dev == 0 or pd.isna(std_dev):
                print(f"Warning: Standard deviation of '{target_col}' is zero or NaN. No outliers removed for this column.")
                continue # Skip this column
            
            mean_val = df_filtered[target_col].mean()
            if pd.isna(mean_val):
                print(f"Warning: Mean of '{target_col}' is NaN. No outliers removed for this column.")
                continue

            z_scores = np.abs((df_filtered[target_col] - mean_val) / std_dev)
            
            # Keep rows where z_score is less than threshold OR where z_score is NaN (i.e., original value was NaN)
            # This ensures we don't drop rows just because the target_col was NaN.
            df_filtered = df_filtered[(z_scores < threshold) | z_scores.isna()]
            
            nr_rows_after = df_filtered.shape[0]
            print(f"  Observations for '{target_col}' before outlier removal: {nr_rows_before}, after: {nr_rows_after}. Removed: {nr_rows_before - nr_rows_after}")

        print("Outlier removal process complete.")
        return df_filtered
        
    def _restrict_sample_logic(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df_processed = df_input.copy()
        begin_date = self.config.get('DATA_BEGIN')
        end_date = self.config.get('DATA_END')
        number_of_banks = self.config.get('RESTRICT_TO_NUMBER_OF_BANKS')
        bank_size_threshold = self.config.get('RESTRICT_TO_BANK_SIZE')
        deposit_ratio_threshold = self.config.get('RESTRICT_TO_MINIMAL_DEPOSIT_RATIO')
        max_change_deposit_ratio = self.config.get('RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO')

        print("\n--- Restricting Sample ---")
        initial_banks = df_processed.index.get_level_values('id').nunique()
        initial_rows = len(df_processed)

        if begin_date is not None:
            df_processed = df_processed[df_processed.index.get_level_values('date') >= begin_date]
            print(f"Filtered to dates >= {begin_date}. Rows remaining: {len(df_processed)}")

        if end_date is not None:
            df_processed = df_processed[df_processed.index.get_level_values('date') <= end_date]
            print(f"Filtered to dates <= {end_date}. Rows remaining: {len(df_processed)}")

        if df_processed.empty:
            print("Warning: DataFrame is empty after date filtering.")
            return df_processed.iloc[0:0]

        if max_change_deposit_ratio is not None:
            if 'deposit_ratio' not in df_processed.columns:
                 print("Warning: 'deposit_ratio' column not found. Cannot filter by max_change_deposit_ratio.")
            else:
                df_processed['abs_change_deposit_ratio'] = df_processed.groupby(level='id')['deposit_ratio'].diff().abs().fillna(0)
                df_processed = df_processed[df_processed['abs_change_deposit_ratio'] <= max_change_deposit_ratio]
                df_processed = df_processed.drop(columns=['abs_change_deposit_ratio'])
                print(f"Filtered by max change in deposit ratio (<= {max_change_deposit_ratio}). Rows remaining: {len(df_processed)}")

        if bank_size_threshold is not None:
            if 'total_assets' not in df_processed.columns:
                print("Warning: 'total_assets' column not found. Cannot filter by bank_size_threshold.")
            else:
                # Calculate mean assets only on the currently processed data
                avg_assets = df_processed.groupby(level='id')['total_assets'].mean()
                ids_meeting_size_criteria = avg_assets[avg_assets >= bank_size_threshold].index
                df_processed = df_processed[df_processed.index.get_level_values('id').isin(ids_meeting_size_criteria)]
                print(f"Filtered to banks with average total assets >= {bank_size_threshold}. Banks: {df_processed.index.get_level_values('id').nunique()}, Rows: {len(df_processed)}")

        if deposit_ratio_threshold is not None:
            if 'deposit_ratio' not in df_processed.columns:
                print("Warning: 'deposit_ratio' column not found. Cannot filter by deposit_ratio_threshold.")
            else:
                # Calculate min deposit ratio only on the currently processed data
                min_ratios_per_bank = df_processed.groupby(level='id')['deposit_ratio'].min()
                ids_meeting_deposit_criteria = min_ratios_per_bank[min_ratios_per_bank >= deposit_ratio_threshold].index
                df_processed = df_processed[df_processed.index.get_level_values('id').isin(ids_meeting_deposit_criteria)]
                print(f"Filtered to banks with minimum deposit ratio >= {deposit_ratio_threshold}. Banks: {df_processed.index.get_level_values('id').nunique()}, Rows: {len(df_processed)}")
        
        if number_of_banks is not None:
            if not isinstance(number_of_banks, int) or number_of_banks <= 0:
                print(f"Warning: 'RESTRICT_TO_NUMBER_OF_BANKS' ({number_of_banks}) is invalid. Skipping this filter.")
            elif initial_banks > number_of_banks: # Apply restriction only if initial count exceeds the limit
                current_bank_ids = df_processed.index.get_level_values('id').unique()
                if len(current_bank_ids) > number_of_banks:
                    # Use a fixed random state for reproducibility
                    sampled_ids = pd.Series(current_bank_ids).sample(n=number_of_banks, random_state=42, replace=False).values
                    df_processed = df_processed[df_processed.index.get_level_values('id').isin(sampled_ids)]
                    print(f"Randomly selected {df_processed.index.get_level_values('id').nunique()} banks.")
                else:
                    print(f"Requested {number_of_banks} banks, but only {len(current_bank_ids)} available. Using all available.")
        
        print(f"Sample restriction complete. Initial banks: {initial_banks}, rows: {initial_rows}. Final banks: {df_processed.index.get_level_values('id').nunique()}, rows: {len(df_processed)}.")
        if df_processed.empty and initial_rows > 0:
            print("Warning: DataFrame became empty after sample restriction.")
        return df_processed # No change here

    def _prepare_data_with_lags_and_target(self, horizon: int, current_target_variable: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[str]]:
        
        # Check
        if self.final_data is None or self.final_data.empty:
            print(f"Base data is not prepared or is empty. Cannot generate data for horizon {horizon}.")
            return None, None, None 

        df = self.final_data.copy()
        df = df.sort_index()

        ar_term_names = []
        # Create AR lags if configured
        if self.include_ar_lags and self.num_lags_to_include > 0:
            for lag_num in range(1, self.num_lags_to_include + 1):
                ar_lag_col_name = f'{current_target_variable}_ar_lag_{lag_num}'
                if current_target_variable in df.columns:
                    df[ar_lag_col_name] = df.groupby(level='id', group_keys=False)[current_target_variable].shift(lag_num)
                    ar_term_names.append(ar_lag_col_name)
                else:
                    print(f"Warning: Target variable '{current_target_variable}' not found in base data. Cannot create AR lags.")
                    return df, [], "" 
        
        # This method now only adds AR lags. The target shifting will happen later.
        # Return an empty string for shifted_target_col as it's not created here.
        shifted_target_col = "" 
        
        return df, ar_term_names, shifted_target_col
    
    def _perform_train_test_split(self, X_full: pd.DataFrame, y_full: pd.Series) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]
    ]:
        """Performs the train/test split based on configuration."""
        X_train, X_test, y_train, y_test = None, None, None, None

        if X_full.empty or y_full.empty or len(X_full) != len(y_full):
            print("Cannot perform train/test split: Input data is empty or mismatched.")
            return None, None, None, None

        if self.train_test_split_dimension == "date":
            unique_dates = X_full.index.get_level_values('date').unique().sort_values()
            if len(unique_dates) < 2:
                print(f"Not enough unique dates ({len(unique_dates)}) for date split. Skipping.")
                return None, None, None, None

            split_date_val: Optional[pd.Timestamp] = None
            if isinstance(self.test_split_config, str):
                try:
                    split_date_val = pd.to_datetime(self.test_split_config)
                    if split_date_val < unique_dates.min() or split_date_val >= unique_dates.max():
                         print(f"Split date '{self.test_split_config}' is outside the data range. Skipping.")
                         return None, None, None, None
                except ValueError:
                    print(f"Invalid date string '{self.test_split_config}' for date split config. Skipping.")
                    return None, None, None, None
            elif isinstance(self.test_split_config, float) and 0.0 < self.test_split_config < 1.0:
                num_test_dates = int(np.ceil(len(unique_dates) * self.test_split_config))
                if num_test_dates == 0: num_test_dates = 1
                split_idx = len(unique_dates) - num_test_dates
                if split_idx <= 0 or split_idx >= len(unique_dates):
                    print(f"Calculated split index ({split_idx}) is invalid for date split config ({self.test_split_config}). Skipping.")
                    return None, None, None, None
                split_date_val = unique_dates[split_idx - 1]
            else:
                print(f"Invalid test_split_config type or value ({self.test_split_config}) for date split. Skipping.")
                return None, None, None, None 

            train_mask = X_full.index.get_level_values('date') <= split_date_val
            test_mask = X_full.index.get_level_values('date') > split_date_val

        elif self.train_test_split_dimension == "id":
            if not (isinstance(self.test_split_config, float) and 0.0 < self.test_split_config < 1.0):
                print(f"Invalid test_split_config type or value ({self.test_split_config}) for ID split. Skipping.")
                return None, None, None, None
            unique_ids = X_full.index.get_level_values('id').unique() 
            if len(unique_ids) < 2:
                print(f"Not enough unique IDs ({len(unique_ids)}) for ID split. Skipping.")
                return None, None, None, None
            num_test_ids = int(np.ceil(len(unique_ids) * self.test_split_config)) 
            if self.test_split_config > 0 and num_test_ids == 0: num_test_ids = 1
            if num_test_ids == 0 or num_test_ids >= len(unique_ids):
                print(f"Calculated number of test IDs ({num_test_ids}) is invalid for ID split config ({self.test_split_config}). Skipping.")
                return None, None, None, None

            gss = GroupShuffleSplit(n_splits=1, test_size=num_test_ids, random_state=42)
            temp_X_reset = X_full.reset_index()
            try:
                train_indices, test_indices = next(gss.split(temp_X_reset, y_full.reset_index(), groups=temp_X_reset['id'])) 
            except ValueError as e:
                print(f"Error during GroupShuffleSplit: {e}. Skipping.")
                return None, None, None, None

            train_mask = X_full.index.isin(temp_X_reset.iloc[train_indices].set_index(['id','date']).index)
            test_mask = X_full.index.isin(temp_X_reset.iloc[test_indices].set_index(['id','date']).index)
        else:
            print(f"Invalid train_test_split_dimension '{self.train_test_split_dimension}'. Must be 'date' or 'id'. Skipping.")
            return None, None, None, None

        X_train = X_full[train_mask]
        X_test = X_full[test_mask]
        y_train = y_full[train_mask]
        y_test = y_full[test_mask]

        if X_train.empty or y_train.empty:
             print("Warning: Train set is empty after split. Cannot proceed.")
             return None, None, None, None

        return X_train, X_test, y_train, y_test

    def _add_feature_lags_internal(self, df_to_lag: pd.DataFrame, num_features_to_lag: List[str], num_lags: int) -> Tuple[pd.DataFrame, List[str]]:
        """Helper to add lags to specified numeric features within a DataFrame."""
        df_out = df_to_lag.copy()
        new_lagged_cols_dict = {}
        lagged_names: List[str] = []

        if num_lags > 0 and not df_out.empty:
            for feat_name in num_features_to_lag: 
                if feat_name in df_out.columns:
                    for lag_i in range(1, num_lags + 1):
                        lag_col = f"{feat_name}_lag_{lag_i}"
                        # Store new columns in a dictionary first
                        new_lagged_cols_dict[lag_col] = df_out.groupby(level='id', group_keys=False)[feat_name].shift(lag_i)
                        lagged_names.append(lag_col)
                else:
                     print(f"Warning: Feature '{feat_name}' not found in DataFrame for lagging.")
            # Concatenate all new lagged columns at once
            if new_lagged_cols_dict:
                df_out = pd.concat([df_out, pd.DataFrame(new_lagged_cols_dict, index=df_out.index)], axis=1)
        return df_out.copy(), lagged_names # Add .copy() to ensure de-fragmentation

    def _combine_features_and_align_target(
        self, 
        X_train_pre_lagged: pd.DataFrame, 
        X_test_pre_lagged: pd.DataFrame,
        y_train_shifted: pd.Series, # Renamed from y_train_full for clarity
        ar_term_names: List[str],
        y_test_shifted: pd.Series, # Renamed from y_test_full for clarity
        horizon: int, 
        numeric_features_for_lags: List[str], 
        categorical_features: List[str],
        num_lags: int
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:

        X_train_with_feature_lags, train_lagged_names = self._add_feature_lags_internal(X_train_pre_lagged, numeric_features_for_lags, num_lags)

        X_test_with_feature_lags = pd.DataFrame(index=X_test_pre_lagged.index)
        test_lagged_names = [] 
        if not X_test_pre_lagged.empty and num_lags > 0 and numeric_features_for_lags:
            common_cols = list(set(X_train_pre_lagged.columns) & set(X_test_pre_lagged.columns)) 
            combined_for_test_feature_lags = pd.concat([X_train_pre_lagged[common_cols], X_test_pre_lagged[common_cols]]).sort_index()

            combined_with_feature_lags, _ = self._add_feature_lags_internal(
                combined_for_test_feature_lags, numeric_features_for_lags, num_lags
            )
            X_test_with_feature_lags = combined_with_feature_lags.loc[X_test_pre_lagged.index]
            test_lagged_names = [
                f"{feat_name}_lag_{lag_i}" for feat_name in numeric_features_for_lags
                for lag_i in range(1, num_lags + 1) if f"{feat_name}_lag_{lag_i}" in X_test_with_feature_lags.columns
            ]

        final_model_feature_names = categorical_features[:] + numeric_features_for_lags[:] # Always include original numeric features
        if num_lags > 0:
            final_model_feature_names.extend(train_lagged_names)
        final_model_feature_names.extend(ar_term_names) 
        final_model_feature_names = sorted([str(item) for item in list(set(final_model_feature_names))])

        final_model_feature_names_existing_train = [col for col in final_model_feature_names if col in X_train_with_feature_lags.columns]
        X_train_main_unscaled = X_train_with_feature_lags[final_model_feature_names_existing_train].copy()

        # --- Process Test Set: Combine X and y, drop NaNs, then split ---
        if not X_test_with_feature_lags.empty: # Only proceed if there's data to process
            final_model_feature_names_existing_test = [
                col for col in final_model_feature_names_existing_train if col in X_test_with_feature_lags.columns
            ]
            X_test_main_unscaled = X_test_with_feature_lags[final_model_feature_names_existing_test].copy()

            # Concatenate features and target, then drop rows with any NaN
            temp_test_df = pd.concat([X_test_main_unscaled, y_test_shifted.rename('target_temp')], axis=1)
            initial_test_rows = len(temp_test_df)
            temp_test_df.dropna(inplace=True)
            if len(temp_test_df) < initial_test_rows:
                print(f"Dropped {initial_test_rows - len(temp_test_df)} rows from test set due to NaNs in features or target.")
            
            # Separate features and target again
            X_test_final = temp_test_df.drop(columns=['target_temp'])
            y_test_final = temp_test_df['target_temp']

        else:
            # If X_test_with_feature_lags was empty, then final X_test and y_test are also empty
            X_test_final = pd.DataFrame(
                columns=final_model_feature_names_existing_train,
                index=X_test_with_feature_lags.index 
            )
            y_test_final = pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=['id', 'date'])) # Empty series

        # --- Process Train Set: Combine X and y, drop NaNs, then split ---
        # Concatenate features and target, then drop rows with any NaN
        temp_train_df = pd.concat([X_train_main_unscaled, y_train_shifted.rename('target_temp')], axis=1)
        initial_train_rows = len(temp_train_df)
        temp_train_df.dropna(inplace=True)
        if len(temp_train_df) < initial_train_rows:
            print(f"Dropped {initial_train_rows - len(temp_train_df)} rows from train set due to NaNs in features or target.")

        # Separate features and target again
        X_train_final = temp_train_df.drop(columns=['target_temp'])
        y_train_final = temp_train_df['target_temp']

        # Final check before returning
        if X_train_final.empty or y_train_final.empty:
            print("Train set empty after combining features and aligning target. Skipping.")
            return None, None, None, None

        return X_train_final, X_test_final, y_train_final, y_test_final

    def _scale_and_encode_features(self, X_train_unscaled: pd.DataFrame, X_test_unscaled: pd.DataFrame) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame]
    ]:
        """Applies StandardScaler to numeric features and OneHotEncoder to categorical features."""
        if X_train_unscaled.empty:
            print("Cannot scale/encode: X_train_unscaled is empty.")
            return None, None

        numeric_cols_for_scaling = X_train_unscaled.select_dtypes(include=np.number).columns.tolist()
        categorical_cols_for_ohe = X_train_unscaled.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if 'does_not_report' in numeric_cols_for_scaling:
            numeric_cols_for_scaling.remove('does_not_report')

        transformers_list = []
        if numeric_cols_for_scaling:
            transformers_list.append(('num', StandardScaler(), numeric_cols_for_scaling))
        if categorical_cols_for_ohe:
            transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None), categorical_cols_for_ohe))

        if not transformers_list: 
            print("Warning: No numeric or categorical features found for scaling/OHE. Returning unscaled/unencoded data.")
            return X_train_unscaled.copy(), X_test_unscaled.copy()
        else:
            preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')
            try:
                X_train_transformed_np = preprocessor.fit_transform(X_train_unscaled)
                transformed_names = preprocessor.get_feature_names_out()
                X_train_final = pd.DataFrame(X_train_transformed_np, columns=transformed_names, index=X_train_unscaled.index)

                if not X_test_unscaled.empty:
                    X_test_transformed_np = preprocessor.transform(X_test_unscaled)
                    X_test_final = pd.DataFrame(X_test_transformed_np, columns=transformed_names, index=X_test_unscaled.index)
                else:
                    X_test_final = pd.DataFrame(columns=transformed_names, index=X_test_unscaled.index)
            except Exception as e:
                 print(f"Error during scaling/OHE: {e}. Returning None for data.")
                 return None, None
        return X_train_final, X_test_final

    def get_horizon_specific_data(self, horizon: int, target_variable: str) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.DataFrame], # No change here
    ]: # Changed X_train_prophet, X_test_prophet to X_train_main_unscaled, X_test_main_unscaled
        print(f"--- Preparing Data for Target: {target_variable}, Horizon H{horizon} ---")

        # Check
        if self.final_data is None or self.final_data.empty:
            print("Base data is not prepared or is empty. Cannot generate data for any horizon.")
            return None, None, None, None, None, None
        
        # 1. Prepare data with AR lags (but NOT shifted target yet)
        df_with_ar_lags, ar_term_names, _ = self._prepare_data_with_lags_and_target(horizon, target_variable)
        
        # Check
        if df_with_ar_lags is None or df_with_ar_lags.empty:
             print(f"Data preparation failed after adding AR lags for {target_variable}, H{horizon}. Skipping.")
             return None, None, None, None, None, None

        df1 = df_with_ar_lags.copy() 
        features = self.final_feature_list[:] # self.final_feature_list is already curated not to include self.target_variable
        numeric_features = [
            f for f in features
            if pd.api.types.is_numeric_dtype(df1[f]) and f in df1.columns # Ensure column exists
        ]
        
        # 2. Define columns to check for NaNs before splitting.
        #    This now includes the ORIGINAL target variable, as the shifted one is not yet created.
        cols_to_check_for_na = [target_variable] 
        # Add feature lags to NA check if they are included
        if self.include_ar_lags and self.num_lags_to_include > 0:
             cols_to_check_for_na.extend([f"{feat}_lag_{lag}" for feat in numeric_features for lag in range(1, self.num_lags_to_include + 1)])

        cols_to_check_for_na.extend(ar_term_names) 
        if self.include_ar_lags and self.num_lags_to_include > 0: # Check if feature lags are to be included
             cols_to_check_for_na.extend([
                 f"{feat}_lag_{lag}" for feat in numeric_features
                 for lag in range(1, self.num_lags_to_include + 1)
             ])
        
        cols_to_check_for_na_existing = [col for col in cols_to_check_for_na if col in df1.columns]
        
        print(f"Dropping NaNs based on subset of columns that will introduce NaNs: {cols_to_check_for_na_existing}")
        rows_before_na_drop = len(df1)
        df_h_filtered = df1.dropna(subset=cols_to_check_for_na_existing).copy()
        
        if df_h_filtered.empty: # Check if empty after NaN drop
            print(f"DataFrame empty for H{horizon} after initial NaN drop. Skipping.")
            return None, None, None, None, None, None

        # 3. Define X_full and y_full (unshifted) for the split
        #    X_full_unshifted contains all features (original + AR lags)
        #    y_full_unshifted contains the ORIGINAL target variable
        X_full_unshifted_cols = self.final_feature_list[:]
        if ar_term_names:
            X_full_unshifted_cols.extend(ar_term_names)

        # Ensure all columns in X_full_unshifted_cols actually exist in df_h_filtered
        X_full_unshifted_cols_existing = [col for col in X_full_unshifted_cols if col in df_h_filtered.columns]
        X_pre_lag_full = df_h_filtered[X_full_unshifted_cols_existing].copy()
        y_full_unshifted = df_h_filtered[target_variable].copy()

        unique_cols_for_X_pre_lag_full = sorted(list(set(str(c) for c in X_full_unshifted_cols_existing)))
        existing_cols_in_df_h_filtered = [c for c in unique_cols_for_X_pre_lag_full if c in df_h_filtered.columns]
        
        if not existing_cols_in_df_h_filtered: # Check if any features exist for X
            print(f"Error: No features found for X_pre_lag_full from df_h_filtered for H{horizon} (cols sought: {unique_cols_for_X_pre_lag_full}). Skipping.")
            return None, None, None, None, None, None
        X_pre_lag_full = df_h_filtered[existing_cols_in_df_h_filtered].copy()

        # 4. Perform the train-test split on unshifted data
        X_train_pre_lag, X_test_pre_lag, y_train_unshifted, y_test_unshifted = self._perform_train_test_split(X_pre_lag_full, y_full_unshifted)

        # 5. Now, shift the target variables *within* their respective splits
        #    This is the crucial step to prevent leakage
        y_train = y_train_unshifted.groupby(level='id', group_keys=False).shift(-horizon)
        y_test = y_test_unshifted.groupby(level='id', group_keys=False).shift(-horizon)

        # Check if the train split resulted in empty data
        # y_train_unshifted is the original target for the train split, used for checking data availability
        if X_train_pre_lag is None or X_train_pre_lag.empty or y_train_unshifted is None or y_train_unshifted.empty:
            print(f"Train set empty after split for H{horizon}. Skipping.")
            return None, None, None, None, None, None

        # Identify categorical features from X_train_pre_lag (which includes original features + AR terms + FE)
        categorical_model_features = [f for f in X_train_pre_lag.columns if not pd.api.types.is_numeric_dtype(X_train_pre_lag[f]) and f not in ar_term_names]
        numeric_features_for_lags_present = [f for f in numeric_features if f in X_train_pre_lag.columns]

        X_train_final_cleaned, X_test_final_cleaned, y_train_final, y_test_final = self._combine_features_and_align_target(
            X_train_pre_lag, X_test_pre_lag, y_train, ar_term_names, y_test, # Pass the newly shifted y_train/y_test
            horizon, numeric_features_for_lags_present, categorical_model_features, self.num_lags_to_include)

        if X_train_final_cleaned is None or X_train_final_cleaned.empty:
            print(f"Train set (unscaled) became empty after combining features for {target_variable}, H{horizon}. Skipping.")
            return None, None, None, None, None, None

        # Pass the already cleaned and aligned X_train_final_cleaned and X_test_final_cleaned to scaling
        X_train_scaled, X_test_scaled = self._scale_and_encode_features(X_train_final_cleaned, X_test_final_cleaned)

        if X_train_scaled is None or X_train_scaled.empty:
             print(f"Train set became empty after scaling/OHE for {target_variable}, H{horizon}. Skipping.")
             return None, None, None, None, None, None

        print(f"Data preparation for H{horizon} complete. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape if X_test_scaled is not None else 'N/A'}")

        return X_train_scaled, X_test_scaled, y_train_final, y_test_final, X_train_final_cleaned, X_test_final_cleaned



    def _correct_structural_breaks_in_total_assets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrects for structural breaks in the 'log_total_assets' time series within a DataFrame.
        Uses the std_dev_threshold from the class configuration.
        """
        df_corrected = df.copy()
        if 'log_total_assets' not in df_corrected.columns:
            print("Warning: 'log_total_assets' column not found or is empty. Skipping structural break correction.")
            return df_corrected

        # Ensure the DataFrame is sorted by id and date for pct_change and shift operations
        df_corrected = df_corrected.sort_index(level=['id', 'date'])

        # Calculate quarter-over-quarter difference for 'log_total_assets' for each bank
        df_corrected['qoq_log_assets_diff'] = df_corrected.groupby(level='id')['log_total_assets'].diff()

        # Calculate the overall standard deviation of these differences
        overall_std_dev_log_diff = df_corrected['qoq_log_assets_diff'].dropna().std()
        print("Structural break correction: Overall mean of log total asset diff:", df_corrected['qoq_log_assets_diff'].mean())
        print("Structural break correction: Overall std dev of log total asset diff:", overall_std_dev_log_diff)
        
        # Determine the break threshold value
        break_threshold_value = self.structural_break_std_dev_threshold * overall_std_dev_log_diff
        print(f"Structural break threshold (abs diff > value): {break_threshold_value:.4f} using threshold: {self.structural_break_std_dev_threshold} std devs")

        # Create a df with bank ids and dates indicating structural breaks
        df_corrected['is_structural_break'] = np.abs(df_corrected['qoq_log_assets_diff']) > break_threshold_value
        # Filter to only those rows where a structural break is detected
        structural_breaks = df_corrected[df_corrected['is_structural_break']].copy()
        if structural_breaks.empty:
            print("No structural breaks detected in log total assets. Returning original DataFrame.")
            df_corrected.drop(columns=['qoq_log_assets_diff', 'is_structural_break'], inplace=True)
            return df_corrected
        print(f"Detected {structural_breaks.shape[0]} structural breaks in log total assets across {structural_breaks.index.get_level_values('id').nunique()} banks.")
        print(f"Mean structural break log diff: {structural_breaks['qoq_log_assets_diff'].mean():.4f}")

        # Iterate over banks in structural_breaks and apply correction
        for bank_id in structural_breaks.index.get_level_values('id').unique():
            bank_breaks = structural_breaks.loc[bank_id].sort_index()
            bank_data = df_corrected.loc[bank_id].sort_index()

            for break_date in bank_breaks.index: # Iterate through each break date for the current bank
                date_str = break_date.strftime('%Y-%m-%d') if pd.notna(break_date) else "NaT"

                # Find index of the period before the break
                try:
                    idx_break_date_in_bank = bank_data.index.get_loc(break_date)
                except KeyError: 
                    print(f"  Skipping break at {date_str} for bank {bank_id}: break_date not in bank_data.index.")
                    continue
                
                if idx_break_date_in_bank == 0: 
                    print(f"  Skipping break at {date_str} for bank {bank_id}: break is at the first observation.")
                    continue
                
                date_before_break_actual = bank_data.index[idx_break_date_in_bank - 1]

                # Get total_assets values for calculating the break_factor
                # These are from bank_data, which reflects df_corrected and thus any prior corrections for this bank
                value_at_break = bank_data.loc[break_date, 'log_total_assets']
                value_before_break = bank_data.loc[date_before_break_actual, 'log_total_assets']

                if pd.notna(value_at_break) and pd.notna(value_before_break) and value_before_break != 0:
                    break_factor = value_at_break / value_before_break
                    
                    # Apply the correction factor to all 'log_total_assets' values *before* the break date for this bank
                    # Correct up to and including date_before_break_actual
                    slice_to_correct_assets = (bank_id, bank_data.index[bank_data.index <= date_before_break_actual])
                    df_corrected.loc[slice_to_correct_assets, 'log_total_assets'] *= break_factor
                    
                    # Recalculate log_total_assets for the corrected period, handling 0 or NaN in total_assets
                    # df_corrected.loc[slice_to_correct_assets, 'log_total_assets'] = np.log(df_corrected.loc[slice_to_correct_assets, 'log_total_assets'].replace(0, np.nan).fillna(1e-9)) # replace 0 with NaN before log, then fill resulting NaN
                    
                    # Update bank_data to reflect changes for subsequent breaks within the same bank
                    bank_data = df_corrected.loc[bank_id].sort_index()
                    # print(f"  Corrected structural break for bank {bank_id} at {date_str} with factor {break_factor:.4f}")
                elif pd.isna(value_at_break) or pd.isna(value_before_break):
                    print(f"  Skipping structural break correction for bank {bank_id} at {date_str}: NaN in total_assets at or before break.")
                elif value_before_break == 0:
                    print(f"  Skipping structural break correction for bank {bank_id} at {date_str}: Zero total_assets before break, cannot calculate factor.")

        df_corrected.drop(columns=['qoq_log_assets_diff'], inplace=True) # Clean up temporary columns

        return df_corrected

    def rectangularize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rectangularizes the input DataFrame.

        Ensures that for each unique 'id', the DataFrame contains all dates
        within the min and max date range observed in the input DataFrame.
        Missing combinations of 'id' and 'date' will be filled with NaNs.

        Args:
            df (pd.DataFrame): Input DataFrame with a MultiIndex ('id', 'date').

        Returns:
            pd.DataFrame: A rectangularized DataFrame.
        """
        unique_ids = df.index.get_level_values('id').unique()
        min_date = df.index.get_level_values('date').min()
        max_date = df.index.get_level_values('date').max()

        all_dates = pd.date_range(min_date, max_date, freq='QS')
        new_index = pd.MultiIndex.from_product([unique_ids, all_dates], names=['id', 'date'])
        return df.reindex(new_index)
