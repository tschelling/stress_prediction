import pandas as pd
import numpy as np
from pyparsing import null_debug_action
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from typing import Any, List, Tuple, Optional, Union, Dict # Ensure Dict is imported

class RegressionDataPreparer:
    
    def __init__(self, initial_df: pd.DataFrame, config: dict[str, any]):
        self.df0 = initial_df.copy()
        self.config = config.copy()

        self.target_variable: str = self.config['TARGET_VARIABLE']
        self.feature_variables: List[str] = self.config['FEATURE_VARIABLES'][:]
        
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
        self.df1_after_initial_cleaning: Optional[pd.DataFrame] = None
        self.df1_after_var_selection: Optional[pd.DataFrame] = None
        self.df1a_after_structural_breaks: Optional[pd.DataFrame] = None
        self.df2_after_missing_value_processing: Optional[pd.DataFrame] = None
        self.df3_after_winsorization: Optional[pd.DataFrame] = None
        self.df4_after_time_fe: Optional[pd.DataFrame] = None
        self.df5_after_bank_fe: Optional[pd.DataFrame] = None
        self.df6_after_fe_dropna: Optional[pd.DataFrame] = None
        self.df7_after_outlier_removal: Optional[pd.DataFrame] = None
        # self.df8_after_sample_restriction will be self.base_data_for_horizons

        self.new_features_added_during_prep: List[str] = [] # To track features like 'does_not_report'
        self.base_data_for_horizons = None
        self.final_feature_list: List[str] = self.feature_variables[:]
        
        self._prepare_initial_data()

    def _prepare_initial_data(self):
        df = self.df0.copy() 
        print(f"\n--- Preparing Initial Data ---")
        print(f"Initial raw data shape: {df.shape}")

        # Select features and target variable
        df = self._select_features_and_target(df)
        self.df1_after_var_selection = df.copy()

        # Process missing values
        df, missing_stats = self._process_missing_values(df)
        self.df2_after_missing_value_processing = df.copy()
        self.data_processing_stats = missing_stats # Store stats
        print(f"After missing value processing (no imputation), data shape: {df.shape}")


        initial_feature_list = self.feature_variables[:]

        for new_feat in self.new_features_added_during_prep:
            if new_feat not in initial_feature_list and new_feat in df.columns:
                initial_feature_list.append(new_feat)
        print(f"Feature list after considering new prep features: {initial_feature_list}")

        if self.include_time_fe:
            if not pd.api.types.is_datetime64_any_dtype(df.index.get_level_values('date')):
                 df.index = df.index.set_levels(pd.to_datetime(df.index.get_level_values('date')), level='date')
                 print("Converted 'date' index level to datetime.")

            df['quarter'] = 'quarter_' + df.index.get_level_values('date').quarter.astype(str)
            if 'quarter' not in initial_feature_list: initial_feature_list.append('quarter')
            self.df4_after_time_fe = df.copy()
            print("Added time fixed effects: 'quarter'.")
        else:
            self.df4_after_time_fe = df.copy() # Store even if no change

        # Include bank fixed effects if configured
        if self.include_bank_fe:
            df['bank_id'] = ['bank_id_' + str(bank_id) for bank_id in df.index.get_level_values('id')]
            if 'bank_id' not in initial_feature_list:
                initial_feature_list = ['bank_id'] + initial_feature_list 
            self.df5_after_bank_fe = df.copy()
            print("Added bank fixed effect: 'bank_id'.")
        else:
            self.df5_after_bank_fe = df.copy() # Store even if no change
        
        self.final_feature_list = initial_feature_list
        
        # Relic, to delete later
        self.df6_after_fe_dropna = None

        if not df.empty:
            df = self._remove_outliers(df, self.target_variable, threshold=self.config.get('OUTLIER_THRESHOLD_TARGET', 3.0))
        else:
            print("DataFrame is empty after FE processing and NaN drop. Skipping outlier removal.")
        self.df7_after_outlier_removal = df.copy()


        if not df.empty:
             df = self._restrict_sample_logic(df)
        else:
            print("DataFrame is empty before sample restriction. Skipping sample restriction.")
        
        # Correct structural breaks in total assets if configured
        if self.correct_structural_breaks_total_assets:
            print("Applying structural break correction for 'total_assets'...")
            df = self._correct_structural_breaks_in_total_assets(df)
        self.df1a_after_structural_breaks = df.copy()

        df = self._winsorize_data(df) 
        self.df3_after_winsorization = df.copy()


        self.base_data_for_horizons = df.copy()
        print(f"Base data prepared. Shape: {self.base_data_for_horizons.shape}")
        print(f"Final features for horizon processing: {self.final_feature_list}")

    def _select_features_and_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects only the target variable and base feature variables."""
        print("Selecting features and target variable...")
        cols_to_select = self.feature_variables[:]
        if self.target_variable not in cols_to_select:
            cols_to_select.append(self.target_variable)
        
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
        
        self.df1_after_initial_cleaning = df.copy()

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

    def _remove_outliers(self, df: pd.DataFrame, target_col: str, threshold: float) -> pd.DataFrame:
        print(f"Removing outliers from the target variable '{target_col}' using z-score threshold {threshold}...")
        nr_rows_before = df.shape[0]
        # Calculate z-scores, handle potential division by zero if std is 0
        std_dev = df[target_col].std()
        if std_dev == 0:
            print(f"Warning: Standard deviation of '{target_col}' is zero. No outliers removed.")
            return df.copy()
        z_scores = np.abs((df[target_col] - df[target_col].mean()) / std_dev)
        df_filtered = df[z_scores < threshold]
        nr_rows_after = df_filtered.shape[0]
        print(f"Observations before outlier removal: {nr_rows_before}, after: {nr_rows_after}. Removed: {nr_rows_before - nr_rows_after}")
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
            else:
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
        return df_processed


    def _prepare_data_with_lags_and_target(self, horizon: int) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[str]]:
        
        # Check
        if self.base_data_for_horizons is None or self.base_data_for_horizons.empty:
            print(f"Base data is not prepared or is empty. Cannot generate data for horizon {horizon}.")
            return None, None, None 

        df = self.base_data_for_horizons.copy()
        df = df.sort_index()

        ar_term_names = []
        # Create AR lags if configured
        if self.include_ar_lags and self.num_lags_to_include > 0:
            for lag_num in range(1, self.num_lags_to_include + 1):
                ar_lag_col_name = f'{self.target_variable}_ar_lag_{lag_num}'
                if self.target_variable in df.columns:
                    df[ar_lag_col_name] = df.groupby(level='id', group_keys=False)[self.target_variable].shift(lag_num)
                    ar_term_names.append(ar_lag_col_name)
                else:
                    print(f"Warning: Target variable '{self.target_variable}' not found in base data. Cannot create AR lags.")
                    return df, [], "" 

        shifted_target_col = f'{self.target_variable}_target_h{horizon}'
        if self.target_variable in df.columns:
             df[shifted_target_col] = df.groupby(level='id', group_keys=False)[self.target_variable].shift(-horizon)
        else:
             print(f"Warning: Target variable '{self.target_variable}' not found in base data. Cannot create shifted target for H{horizon}.")
             return df, ar_term_names, "" # Return empty string for shifted_target_col if failed
        
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
        lagged_names: List[str] = []
        if num_lags > 0 and not df_out.empty:
            for feat_name in num_features_to_lag: 
                if feat_name in df_out.columns:
                    for lag_i in range(1, num_lags + 1):
                        lag_col = f"{feat_name}_lag_{lag_i}"
                        df_out[lag_col] = df_out.groupby(level='id', group_keys=False)[feat_name].shift(lag_i)
                        lagged_names.append(lag_col)
                else:
                     print(f"Warning: Feature '{feat_name}' not found in DataFrame for lagging.")
        return df_out, lagged_names

    def _combine_features_and_align_target(
        self, 
        X_train_pre_lagged: pd.DataFrame, 
        X_test_pre_lagged: pd.DataFrame,  
        y_train_full: pd.Series, 
        ar_term_names: List[str],
        y_test_full: pd.Series, 
        horizon: int, 
        numeric_features_for_lags: List[str], 
        categorical_features: List[str],
        num_lags: int
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:

        X_train_with_feature_lags, train_lagged_names = self._add_feature_lags_internal(
            X_train_pre_lagged, numeric_features_for_lags, num_lags
        )

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

        final_model_feature_names = categorical_features[:]
        if num_lags > 0:
            final_model_feature_names.extend(train_lagged_names)
        else: 
            final_model_feature_names.extend(numeric_features_for_lags)
        final_model_feature_names.extend(ar_term_names) 
        final_model_feature_names = sorted([str(item) for item in list(set(final_model_feature_names))])

        final_model_feature_names_existing_train = [col for col in final_model_feature_names if col in X_train_with_feature_lags.columns]
        X_train_main_unscaled = X_train_with_feature_lags[final_model_feature_names_existing_train].copy()

        if not X_test_with_feature_lags.empty: 
            final_model_feature_names_existing_test = [
                col for col in final_model_feature_names_existing_train if col in X_test_with_feature_lags.columns
            ]
            X_test_main_unscaled = X_test_with_feature_lags[final_model_feature_names_existing_test].copy()
            initial_test_rows = len(X_test_main_unscaled)
            X_test_main_unscaled.dropna(inplace=True)
            if len(X_test_main_unscaled) < initial_test_rows:
                 print(f"Dropped {initial_test_rows - len(X_test_main_unscaled)} rows from test set due to NaNs after lagging/AR terms.")
            
            if not X_test_main_unscaled.empty:
                y_test = y_test_full.loc[y_test_full.index.intersection(X_test_main_unscaled.index)].copy() 
                X_test_main_unscaled = X_test_main_unscaled.loc[y_test.index]
            else: 
                y_test = pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=['id', 'date']))
        else: 
            X_test_main_unscaled = pd.DataFrame(
                columns=final_model_feature_names_existing_train,
                index=X_test_with_feature_lags.index 
            )
            y_test = pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=['id', 'date']))
        
        initial_train_rows = len(X_train_main_unscaled)
        X_train_main_unscaled.dropna(inplace=True)
        if len(X_train_main_unscaled) < initial_train_rows:
             print(f"Dropped {initial_train_rows - len(X_train_main_unscaled)} rows from train set due to NaNs after lagging/AR terms.")

        y_train = y_train_full.loc[y_train_full.index.intersection(X_train_main_unscaled.index)].copy() 
        X_train_main_unscaled = X_train_main_unscaled.loc[y_train.index] # Re-align X_train to y_train after y_train is potentially reduced
        
        if not X_test_main_unscaled.empty and not y_test.empty: # Ensure X_test is re-aligned if y_test changed
            X_test_main_unscaled = X_test_main_unscaled.loc[y_test.index]


        if X_train_main_unscaled.empty or y_train.empty:
            print("Train set empty after combining features and aligning target. Skipping.")
            return None, None, None, None

        return X_train_main_unscaled, X_test_main_unscaled, y_train, y_test 

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


    def get_horizon_specific_data(self, horizon: int) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.DataFrame],
    ]: # Changed X_train_prophet, X_test_prophet to X_train_main_unscaled, X_test_main_unscaled
        print(f"--- Preparing Data for Horizon H{horizon} ---")

        # Check
        if self.base_data_for_horizons is None or self.base_data_for_horizons.empty:
            print("Base data is not prepared or is empty. Cannot generate data for any horizon.")
            return None, None, None, None, None, None

        df, ar_term_names, shifted_target_col = self._prepare_data_with_lags_and_target(horizon)
        
        # Check
        if df is None or not shifted_target_col:
             print(f"Data preparation failed after adding lags/target for H{horizon}. Skipping.")
             return None, None, None, None, None, None

        df1 = df.copy() 
        features = [
            f for f in self.final_feature_list
            if f != self.target_variable 
        ]
        numeric_features = [
            f for f in features
            if pd.api.types.is_numeric_dtype(df1[f]) and f in df1.columns # Ensure column exists
        ]
        cols_to_check_for_na = [shifted_target_col] 
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
        
        if df_h_filtered.empty:
            print(f"DataFrame empty for H{horizon} after initial NaN drop. Skipping.")
            return None, None, None, None, None, None

        cols_for_X_pre_lag_full = self.final_feature_list[:]
        if ar_term_names: 
            cols_for_X_pre_lag_full.extend(ar_term_names) 
        
        unique_cols_for_X_pre_lag_full = sorted(list(set(str(c) for c in cols_for_X_pre_lag_full)))
        existing_cols_in_df_h_filtered = [c for c in unique_cols_for_X_pre_lag_full if c in df_h_filtered.columns]
        
        if not existing_cols_in_df_h_filtered:
            print(f"Error: No features found for X_pre_lag_full from df_h_filtered for H{horizon} (cols sought: {unique_cols_for_X_pre_lag_full}). Skipping.")
            return None, None, None, None, None, None
        X_pre_lag_full = df_h_filtered[existing_cols_in_df_h_filtered].copy()

        if shifted_target_col not in df_h_filtered.columns:
             print(f"Error: Shifted target column '{shifted_target_col}' not found after processing for H{horizon}. Skipping.")
             return None, None, None, None, None, None
        y_full = df_h_filtered[shifted_target_col]

        if X_pre_lag_full.empty or y_full.empty or len(X_pre_lag_full) != len(y_full):
            print(f"X_pre_lag_full or y_full empty or mismatched for H{horizon} before split. Skipping.")
            return None, None, None, None, None, None

        X_train_pre_lag, X_test_pre_lag, y_train_full, y_test_full = self._perform_train_test_split(X_pre_lag_full, y_full)


        if X_train_pre_lag is None or X_train_pre_lag.empty or y_train_full is None or y_train_full.empty:
            print(f"Train set empty after split for H{horizon}. Skipping.")
            return None, None, None, None, None, None
        
        # Identify categorical features from X_train_pre_lag (which includes original features + AR terms + FE)
        # These are features that are not numeric and not AR terms.
        categorical_model_features = [
            f for f in X_train_pre_lag.columns 
            if not pd.api.types.is_numeric_dtype(X_train_pre_lag[f]) and f not in ar_term_names
        ]
        # Also ensure numeric_features_for_lags only contains columns present in X_train_pre_lag
        numeric_features_for_lags_present = [f for f in numeric_features if f in X_train_pre_lag.columns]


        X_train_main_unscaled, X_test_main_unscaled, y_train, y_test = self._combine_features_and_align_target(
            X_train_pre_lag, X_test_pre_lag, y_train_full, ar_term_names, y_test_full,
            horizon, numeric_features_for_lags_present, categorical_model_features, self.num_lags_to_include) # Pass present numeric features

        if X_train_main_unscaled is None or X_train_main_unscaled.empty:
            print(f"Train set (unscaled) became empty after combining features for H{horizon}. Skipping.")
            return None, None, None, None, None, None


        X_train_final, X_test_final = self._scale_and_encode_features(X_train_main_unscaled, X_test_main_unscaled)

        if X_train_final is None or X_train_final.empty:
             print(f"Train set became empty after scaling/OHE for H{horizon}. Skipping.")
             return None, None, None, None, None, None

        print(f"Data preparation for H{horizon} complete. Train shape: {X_train_final.shape}, Test shape: {X_test_final.shape if X_test_final is not None else 'N/A'}")

        return X_train_final, X_test_final, y_train, y_test, X_train_main_unscaled, X_test_main_unscaled

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
