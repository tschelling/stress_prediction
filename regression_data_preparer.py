import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from typing import Any, List, Tuple, Optional, Union # Import necessary types

class RegressionDataPreparer:
    def __init__(self, initial_df: pd.DataFrame, config: dict[str, any]): # Now 'Any' is defined
        self.initial_df = initial_df.copy()
        self.config = config.copy()

        self.target_variable: str = self.config['TARGET_VARIABLE']
        self.feature_variables_base: List[str] = self.config['FEATURE_VARIABLES'][:]
        
        self.include_time_fe: bool = self.config.get('INCLUDE_TIME_FE', False)
        self.include_bank_fe: bool = self.config.get('INCLUDE_BANK_FE', False)
        self.include_ar_lags: bool = self.config.get('INCLUDE_AUTOREGRESSIVE_LAGS', True)
        self.num_lags_to_include: int = self.config.get('NUMBER_OF_LAGS_TO_INCLUDE', 0)

        self.train_test_split_dimension = self.config['TRAIN_TEST_SPLIT_DIMENSION']
        self.test_split_config = self.config['TEST_SPLIT']

        self.base_data_for_horizons = None
        self.final_feature_list_for_horizon_processing: List[str] = self.feature_variables_base[:]

        self._prepare_initial_data()

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
        
    def _remove_banks_with_few_observations(self, df: pd.DataFrame, min_observations: int) -> pd.DataFrame:
        print(f"Removing banks with fewer than {min_observations} observations...")
        nr_rows_before = df.shape[0]
        bank_counts = df.index.get_level_values('id').value_counts()
        banks_to_remove = bank_counts[bank_counts < min_observations].index
        df_filtered = df[~df.index.get_level_values('id').isin(banks_to_remove)]
        nr_rows_after = df_filtered.shape[0]
        print(f"Observations before removing banks with few obs: {nr_rows_before}, after: {nr_rows_after}. Removed: {nr_rows_before - nr_rows_after}")
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

    def _prepare_initial_data(self):
        df = self.initial_df.copy() # Work on a copy to avoid modifying the original initial_df
        print(f"\n--- Preparing Initial Data ---")
        print(f"Initial raw data shape: {df.shape}")

        current_feature_list = self.feature_variables_base[:]

        if self.include_time_fe:
            # Ensure 'date' is datetime type for quarter/year extraction
            if not pd.api.types.is_datetime64_any_dtype(df.index.get_level_values('date')):
                 df.index = df.index.set_levels(pd.to_datetime(df.index.get_level_values('date')), level='date')
                 print("Converted 'date' index level to datetime.")

            df['quarter'] = 'quarter_' + df.index.get_level_values('date').quarter.astype(str)
            #df['year'] = 'year_' + df.index.get_level_values('date').year.astype(str)
            if 'quarter' not in current_feature_list: current_feature_list.append('quarter')
            #if 'year' not in current_feature_list: current_feature_list.append('year')
            print("Added time fixed effects: 'quarter'.")

        if self.include_bank_fe:
            # Use a list comprehension for robust string concatenation.
            # This avoids potential UFuncTypeErrors with NumPy string dtypes.
            df['bank_id'] = ['bank_id_' + str(bank_id) for bank_id in df.index.get_level_values('id')]

            if 'bank_id' not in current_feature_list:
                current_feature_list = ['bank_id'] + current_feature_list # Prepend
            print("Added bank fixed effect: 'bank_id'.")
        
        self.final_feature_list_for_horizon_processing = current_feature_list

        # Ensure all columns for dropna actually exist in df before subsetting
        cols_for_dropna = [col for col in self.final_feature_list_for_horizon_processing + [self.target_variable] if col in df.columns]
        
        if not cols_for_dropna:
             print("Warning: No valid columns found for initial NaN drop. Skipping initial NaN drop.")
        else:
            print(f"Dropping NaNs based on subset: {cols_for_dropna}")
            rows_before_na_drop = len(df)
            df = df.dropna(subset=cols_for_dropna).copy()
            print(f"Rows before NaN drop: {rows_before_na_drop}, after: {len(df)}. Removed: {rows_before_na_drop - len(df)}")

        if not df.empty:
            df = self._remove_outliers(df, self.target_variable, threshold=self.config.get('OUTLIER_THRESHOLD_TARGET', 3.0))
            if not df.empty:
                 df = self._remove_banks_with_few_observations(df, min_observations=self.config.get('MIN_OBS_PER_BANK', 10))
            else:
                 print("DataFrame became empty after outlier removal. Skipping bank observation removal.")
        else:
            print("DataFrame is empty after initial NaN drop. Skipping outlier and bank observation removal.")

        if not df.empty:
             df = self._restrict_sample_logic(df)
        else:
            print("DataFrame is empty before sample restriction. Skipping sample restriction.")
        
        self.base_data_for_horizons = df.copy()
        print(f"Base data prepared. Shape: {self.base_data_for_horizons.shape}")
        print(f"Final features for horizon processing: {self.final_feature_list_for_horizon_processing}")
    
    def _prepare_data_with_lags_and_target(self, horizon: int) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[str]]:

        if self.base_data_for_horizons is None or self.base_data_for_horizons.empty:
            print(f"Base data is not prepared or is empty. Cannot generate data for horizon {horizon}.")
            return None, None, None # Corrected to return three None values

        df_h = self.base_data_for_horizons.copy()
        df_h = df_h.sort_index()

        ar_term_names = []
        if self.include_ar_lags and self.num_lags_to_include > 0:
            for lag_num in range(1, self.num_lags_to_include + 1):
                ar_lag_col_name = f'{self.target_variable}_ar_lag_{lag_num}'
                # Ensure target variable exists before trying to shift
                # This check is redundant if _prepare_initial_data ensures target exists, but kept for safety
                if self.target_variable in df_h.columns:
                    # Use group_keys=False to prevent adding group keys to the index
                    df_h[ar_lag_col_name] = df_h.groupby(level='id', group_keys=False)[self.target_variable].shift(lag_num)
                    ar_term_names.append(ar_lag_col_name)
                else:
                    print(f"Warning: Target variable '{self.target_variable}' not found in base data. Cannot create AR lags.")
                    # If target is missing, we cannot create any AR lags, so return the original df and empty list
                    return df_h, [], "" # Return df_h, empty ar_term_names, empty shifted_target_col

        # Call _add_shifted_target to actually perform the shift and get the column name
        df_h, shifted_target_col = self._add_shifted_target(df_h, horizon)
        if not shifted_target_col: # If _add_shifted_target failed (e.g., target missing)
            print(f"Failed to add shifted target for horizon {horizon}.")
            # Return df_h as is, with any AR terms added, but no shifted target
            return df_h, ar_term_names, "" # Return current df_h, ar_terms, and empty shifted_target_col

        return df_h, ar_term_names, shifted_target_col
    
    def _add_shifted_target(self, df_h: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, str]:
        """Adds the shifted target variable for the given horizon."""
        shifted_target_col = f'{self.target_variable}_target_h{horizon}'
        # Ensure target variable exists before trying to shift
        if self.target_variable in df_h.columns:
             # Use group_keys=False to prevent adding group keys to the index
             df_h[shifted_target_col] = df_h.groupby(level='id', group_keys=False)[self.target_variable].shift(-horizon)
        else:
             print(f"Warning: Target variable '{self.target_variable}' not found in base data. Cannot create shifted target.")
             # Return original df and an empty string if target is missing
             return df_h, ""

        return df_h, shifted_target_col

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
                return None, None, None, None # Invalid split config

            train_mask = X_full.index.get_level_values('date') <= split_date_val
            test_mask = X_full.index.get_level_values('date') > split_date_val

        elif self.train_test_split_dimension == "id":
            if not (isinstance(self.test_split_config, float) and 0.0 < self.test_split_config < 1.0):
                print(f"Invalid test_split_config type or value ({self.test_split_config}) for ID split. Skipping.")
                return None, None, None, None
            unique_ids = X_full.index.get_level_values('id').unique() # type: ignore
            if len(unique_ids) < 2:
                print(f"Not enough unique IDs ({len(unique_ids)}) for ID split. Skipping.")
                return None, None, None, None
            num_test_ids = int(np.ceil(len(unique_ids) * self.test_split_config)) # type: ignore
            if self.test_split_config > 0 and num_test_ids == 0: num_test_ids = 1
            if num_test_ids == 0 or num_test_ids >= len(unique_ids):
                print(f"Calculated number of test IDs ({num_test_ids}) is invalid for ID split config ({self.test_split_config}). Skipping.")
                return None, None, None, None

            gss = GroupShuffleSplit(n_splits=1, test_size=num_test_ids, random_state=42)
            # Need to reset index for GSS groups argument
            temp_X_reset = X_full.reset_index()
            try:
                # Pass temp_X_reset and y_full.reset_index() to split, but use temp_X_reset['id'] for groups
                train_indices, test_indices = next(gss.split(temp_X_reset, y_full.reset_index(), groups=temp_X_reset['id'])) # type: ignore
            except ValueError as e:
                print(f"Error during GroupShuffleSplit: {e}. Skipping.")
                return None, None, None, None

            # Create masks from indices
            train_mask = X_full.index.isin(temp_X_reset.iloc[train_indices].set_index(['id','date']).index)
            test_mask = X_full.index.isin(temp_X_reset.iloc[test_indices].set_index(['id','date']).index)
        else:
            print(f"Invalid train_test_split_dimension '{self.train_test_split_dimension}'. Must be 'date' or 'id'. Skipping.")
            return None, None, None, None

        # Return the split dataframes/series
        X_train = X_full[train_mask]
        X_test = X_full[test_mask]
        y_train = y_full[train_mask]
        y_test = y_full[test_mask]

        # Check if train set is empty after split
        if X_train.empty or y_train.empty:
             print("Warning: Train set is empty after split. Cannot proceed.")
             return None, None, None, None

        return X_train, X_test, y_train, y_test

    def _add_feature_lags_internal(self, df_to_lag: pd.DataFrame, num_features_to_lag: List[str], num_lags: int) -> Tuple[pd.DataFrame, List[str]]:
        """Helper to add lags to specified numeric features within a DataFrame."""
        df_out = df_to_lag.copy()
        lagged_names: List[str] = []
        if num_lags > 0 and not df_out.empty:
            for feat_name in num_features_to_lag: # type: ignore
                # Ensure feature exists before lagging
                if feat_name in df_out.columns:
                    for lag_i in range(1, num_lags + 1):
                        lag_col = f"{feat_name}_lag_{lag_i}"
                        # Use group_keys=False to prevent adding group keys to the index
                        df_out[lag_col] = df_out.groupby(level='id', group_keys=False)[feat_name].shift(lag_i)
                        lagged_names.append(lag_col)
                else:
                     print(f"Warning: Feature '{feat_name}' not found in DataFrame for lagging.")
        return df_out, lagged_names

    def _combine_features_and_align_target(
        self, # Keep self as it's a method
        X_train_pre_lagged: pd.DataFrame, # This now includes AR terms and is already NaN-filtered based on target/AR/feature lags
        X_test_pre_lagged: pd.DataFrame,  # This now includes AR terms and is already NaN-filtered based on target/AR/feature lags
        y_train_full: pd.Series, # This is already aligned with X_train_pre_lagged
        ar_term_names: List[str],
        y_test_full: pd.Series, # Moved y_test_full here
        horizon: int, # Add horizon as an argument
        numeric_features_for_lags: List[str], # Keep this argument
        categorical_features: List[str],
        num_lags: int
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
        """
        Adds feature lags and AR terms, selects final columns, drops resulting NaNs,
        and aligns target variables.
        """
        # X_train_pre_lagged and X_test_pre_lagged already contain AR terms
        # and have been filtered for NaNs from shifted target, AR terms, AND feature lags.
        # y_train_full and y_test_full are already aligned with these.

        # Add feature lags to train set
        X_train_with_feature_lags, train_lagged_names = self._add_feature_lags_internal(
            X_train_pre_lagged, numeric_features_for_lags, num_lags
        )

        # Add feature lags to test set (requires combining train and test for correct shifting)
        X_test_with_feature_lags = pd.DataFrame(index=X_test_pre_lagged.index) # Init empty
        test_lagged_names = [] # Keep track of test lagged names for column alignment
        if not X_test_pre_lagged.empty and num_lags > 0 and numeric_features_for_lags:
            # For test lags, concat train_pre_lagged and test_pre_lagged to avoid lookahead bias within test but allow history from train
            # Ensure columns match before concat
            common_cols = list(set(X_train_pre_lagged.columns) & set(X_test_pre_lagged.columns)) # Should be original features + AR terms
            combined_for_test_feature_lags = pd.concat([X_train_pre_lagged[common_cols], X_test_pre_lagged[common_cols]]).sort_index()

            combined_with_feature_lags, _ = self._add_feature_lags_internal(
                combined_for_test_feature_lags, numeric_features_for_lags, num_lags
            )
            X_test_with_feature_lags = combined_with_feature_lags.loc[X_test_pre_lagged.index]

            # Generate test lagged names based on the columns that were actually lagged
            test_lagged_names = [
                f"{feat_name}_lag_{lag_i}" for feat_name in numeric_features_for_lags
                for lag_i in range(1, num_lags + 1) if f"{feat_name}_lag_{lag_i}" in X_test_with_feature_lags.columns
            ]

        # Define final features for main models
        final_model_feature_names = categorical_features[:]
        if num_lags > 0:
            # Use the names generated from the train set lagging
            final_model_feature_names.extend(train_lagged_names)
        else: # No feature lags, use original numeric features
            final_model_feature_names.extend(numeric_features_for_lags)

        # Add AR terms (if any) to this list
        final_model_feature_names.extend(ar_term_names) # ar_term_names are already in X_train_pre_lagged/X_test_pre_lagged

        # Ensure all elements are strings before sorting to avoid TypeError
        final_model_feature_names = sorted([str(item) for item in list(set(final_model_feature_names))])

        # Ensure only final_model_feature_names are present and in order for the train set
        final_model_feature_names_existing_train = [col for col in final_model_feature_names if col in X_train_with_feature_lags.columns]
        X_train_main_unscaled = X_train_with_feature_lags[final_model_feature_names_existing_train].copy()

        # Ensure only final_model_feature_names are present and in order
        if not X_test_with_feature_lags.empty: # Process test set only if it's not empty
            # Filter test columns to match the final train columns
            # Use X_test_with_feature_lags.columns as X_test_main_unscaled is not fully defined for this check yet
            final_model_feature_names_existing_test = [
                col for col in final_model_feature_names_existing_train if col in X_test_with_feature_lags.columns
            ]
            X_test_main_unscaled = X_test_with_feature_lags[final_model_feature_names_existing_test].copy()

            # NaNs from feature lags should ideally be minimal or zero here because we dropped based on *all* potential lags before the split.
            # However, a final dropna is a safeguard.
            initial_test_rows = len(X_test_main_unscaled)
            X_test_main_unscaled.dropna(inplace=True)
            if len(X_test_main_unscaled) < initial_test_rows:
                 print(f"Dropped {initial_test_rows - len(X_test_main_unscaled)} rows from test set due to NaNs after lagging/AR terms.")
            
            # Ensure y_test index is a subset of X_test_main_unscaled index before loc
            # Also, handle if X_test_main_unscaled became empty after dropna
            if not X_test_main_unscaled.empty:
                y_test = y_test_full.loc[y_test_full.index.intersection(X_test_main_unscaled.index)].copy() # Use .copy() for safety
                # Re-align X_test to the potentially smaller y_test index
                X_test_main_unscaled = X_test_main_unscaled.loc[y_test.index]
            else: # X_test_main_unscaled is now empty after dropna
                y_test = pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=['id', 'date']))
                # X_test_main_unscaled is already empty, its columns are correct from its creation

        else: # X_test_with_feature_lags was empty
            # Ensure empty test df has correct columns matching train
            # Use X_test_with_feature_lags.index as X_test_main_unscaled.index would be undefined here.
            X_test_main_unscaled = pd.DataFrame(
                columns=final_model_feature_names_existing_train,
                index=X_test_with_feature_lags.index 
            )
            # Ensure y_test is empty if X_test is empty
            y_test = pd.Series(dtype='float64', index=pd.MultiIndex.from_tuples([], names=['id', 'date']))
        
        # NaNs from feature lags should ideally be minimal or zero in train set here
        # because we dropped based on *all* potential lags before the split.
        # However, a final dropna is a safeguard.
        initial_train_rows = len(X_train_main_unscaled)
        X_train_main_unscaled.dropna(inplace=True)
        if len(X_train_main_unscaled) < initial_train_rows:
             print(f"Dropped {initial_train_rows - len(X_train_main_unscaled)} rows from train set due to NaNs after lagging/AR terms.")

        # Re-align y_train and y_test to the potentially smaller X_train/X_test indices after the final dropna
        # This step is crucial if any rows were dropped in the final dropna calls above.
        y_train = y_train_full.loc[y_train_full.index.intersection(X_train_main_unscaled.index)].copy() # Use .copy() for safety
        # X_train_main_unscaled is already aligned to y_train.index by the dropna/loc above
        X_test_main_unscaled = X_test_main_unscaled.loc[y_test.index] # Corrected variable name

        if X_train_main_unscaled.empty or y_train.empty:
            print("Train set empty after combining features and aligning target. Skipping.")
            return None, None, None, None

        return X_train_main_unscaled, X_test_main_unscaled, y_train, y_test # type: ignore

    def _scale_and_encode_features(self, X_train_unscaled: pd.DataFrame, X_test_unscaled: pd.DataFrame) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame]
    ]:
        """Applies StandardScaler to numeric features and OneHotEncoder to categorical features."""
        if X_train_unscaled.empty:
            print("Cannot scale/encode: X_train_unscaled is empty.")
            return None, None

        # Identify numeric and categorical columns based on the train set
        numeric_cols_for_scaling = X_train_unscaled.select_dtypes(include=np.number).columns.tolist()
        categorical_cols_for_ohe = X_train_unscaled.select_dtypes(include=['object', 'category']).columns.tolist()

        transformers_list = []
        if numeric_cols_for_scaling:
            transformers_list.append(('num', StandardScaler(), numeric_cols_for_scaling))
        if categorical_cols_for_ohe:
            # Use handle_unknown='ignore' and drop=None for robustness
            transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None), categorical_cols_for_ohe))

        if not transformers_list: # No numeric or categorical features to transform
            print("Warning: No numeric or categorical features found for scaling/OHE. Returning unscaled/unencoded data.")
            return X_train_unscaled.copy(), X_test_unscaled.copy()
        else:
            # Use remainder='drop' as we've already selected the final columns
            preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')

            try:
                X_train_transformed_np = preprocessor.fit_transform(X_train_unscaled)
                transformed_names = preprocessor.get_feature_names_out()
                X_train_final = pd.DataFrame(X_train_transformed_np, columns=transformed_names, index=X_train_unscaled.index)

                if not X_test_unscaled.empty:
                    X_test_transformed_np = preprocessor.transform(X_test_unscaled)
                    X_test_final = pd.DataFrame(X_test_transformed_np, columns=transformed_names, index=X_test_unscaled.index)
                else:
                    # Create an empty test DataFrame with the correct columns
                    X_test_final = pd.DataFrame(columns=transformed_names, index=X_test_unscaled.index)
            except Exception as e:
                 print(f"Error during scaling/OHE: {e}. Returning None for data.")
                 return None, None

        return X_train_final, X_test_final

    def get_horizon_specific_data(self, horizon: int) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series],
    ]:
        # Removed Prophet dataframes from return signature
        """
        Orchestrates the data preparation steps for a specific forecast horizon,
        returning scaled/encoded and unscaled train/test data, and the targets.
        """
        print(f"--- Preparing Data for Horizon H{horizon} ---")

        if self.base_data_for_horizons is None or self.base_data_for_horizons.empty:
            print("Base data is not prepared or is empty. Cannot generate data for any horizon.")
            return None, None, None, None, None, None

        # Start with a fresh copy of the base data for this horizon
        df_h = self.base_data_for_horizons.copy()
        
        # Step 1: Add AR lags and Shifted Target
        df_h_with_lags_target, ar_term_names, shifted_target_col = self._prepare_data_with_lags_and_target(horizon)

        if df_h_with_lags_target is None or not shifted_target_col:
             print(f"Data preparation failed after adding lags/target for H{horizon}. Skipping.")
             return None, None, None, None

        df_h_processed = df_h_with_lags_target # Use the DataFrame with lags and shifted target

        # Identify numeric features for which lags will be created *after* the split
        # These are the original numeric features, excluding the target and any added FE columns
        original_feature_candidates = [
            f for f in self.final_feature_list_for_horizon_processing
            if f != self.target_variable # Target is handled separately
        ]
        numeric_features_for_lags = [
            f for f in original_feature_candidates
            if pd.api.types.is_numeric_dtype(df_h_processed[f])
        ]

        # Identify ALL columns that will introduce NaNs due to shifting/lagging
        cols_to_check_for_na = [shifted_target_col] # Shifted target always introduces NaNs
        cols_to_check_for_na.extend(ar_term_names) # AR terms introduce NaNs
        # Add names of columns that will be created by feature lagging
        if self.include_ar_lags and self.num_lags_to_include > 0:
             cols_to_check_for_na.extend([
                 f"{feat}_lag_{lag}" for feat in numeric_features_for_lags
                 for lag in range(1, self.num_lags_to_include + 1)
             ])
        
        # Ensure all columns in cols_to_check_for_na actually exist in df_h_processed before subsetting
        cols_to_check_for_na_existing = [col for col in cols_to_check_for_na if col in df_h_processed.columns]

        # Step 2: Drop NaNs based on ALL columns that will introduce them (shifted target, AR, feature lags)
        print(f"Dropping NaNs based on subset of columns that will introduce NaNs: {cols_to_check_for_na_existing}")
        rows_before_na_drop = len(df_h_processed)
        df_h_filtered = df_h_processed.dropna(subset=cols_to_check_for_na_existing).copy()
        
        if df_h_filtered.empty:
            print(f"DataFrame empty for H{horizon} after initial NaN drop. Skipping.")
            return None, None, None, None, None, None

        # Extract full X and y before feature lagging and final NaN drop
        # X_pre_lag_full will contain original features, AR terms, and potentially FE columns.
        cols_for_X_pre_lag_full = self.final_feature_list_for_horizon_processing[:]
        if ar_term_names: # ar_term_names comes from _prepare_data_with_lags_and_target
            cols_for_X_pre_lag_full.extend(ar_term_names) # These columns are already in df_h_filtered
        
        # Ensure all columns exist in df_h_filtered before selection and are unique
        unique_cols_for_X_pre_lag_full = sorted(list(set(str(c) for c in cols_for_X_pre_lag_full)))
        existing_cols_in_df_h_filtered = [c for c in unique_cols_for_X_pre_lag_full if c in df_h_filtered.columns]
        
        if not existing_cols_in_df_h_filtered:
            print(f"Error: No features found for X_pre_lag_full from df_h_filtered for H{horizon} (cols sought: {unique_cols_for_X_pre_lag_full}). Skipping.")
            return None, None, None, None, None, None
        X_pre_lag_full = df_h_filtered[existing_cols_in_df_h_filtered].copy()

        # Ensure shifted target column exists before accessing
        if shifted_target_col not in df_h_filtered.columns:
             print(f"Error: Shifted target column '{shifted_target_col}' not found after processing for H{horizon}. Skipping.")
             return None, None, None, None, None, None

        y_full = df_h_filtered[shifted_target_col]

        if X_pre_lag_full.empty or y_full.empty or len(X_pre_lag_full) != len(y_full):
            print(f"X_pre_lag_full or y_full empty or mismatched for H{horizon} before split. Skipping.")
            return None, None, None, None, None, None

        # Step 3: Perform Train/Test Split on data *before* feature lagging
        X_train_pre_lag, X_test_pre_lag, y_train_full, y_test_full = self._perform_train_test_split(X_pre_lag_full, y_full)

        if X_train_pre_lag is None or X_train_pre_lag.empty or y_train_full is None or y_train_full.empty:
            print(f"Train set empty after split for H{horizon}. Skipping.")
            return None, None, None, None, None, None

        # Identify numeric (for lagging) and categorical features from X_train_pre_lag.
        # We already identified `numeric_features_for_lags` before the split.
        # Now identify categorical features from the split data.
        # These should be the original categorical features + any categorical FE columns ('quarter', 'year', 'bank_id').
        categorical_model_features = [ # Original categorical features
            f for f in X_train_pre_lag.columns # Check columns in the split data
            if f not in ar_term_names and # Exclude AR terms
               f not in numeric_features_for_lags and # Exclude numeric features that will be lagged
               f not in self.feature_variables_base and # Exclude original base features already covered
               f != self.target_variable # self.target_variable should not be in X_train_pre_lag anyway
        ]

        # Step 5: Add Feature Lags, Combine Features (Original + Feature Lags + AR), Drop NaNs, Align Target
        # Corrected argument order: y_train_full, ar_term_names, y_test_full
        X_train_main_unscaled, X_test_main_unscaled, y_train, y_test = self._combine_features_and_align_target(
            X_train_pre_lag, X_test_pre_lag, y_train_full, ar_term_names, y_test_full,
            horizon, numeric_features_for_lags, categorical_model_features, self.num_lags_to_include)

        # Step 7: Scale and Encode Features for main models
        X_train_final, X_test_final = self._scale_and_encode_features(X_train_main_unscaled, X_test_main_unscaled)

        if X_train_final is None or X_train_final.empty:
             print(f"Train set became empty after scaling/OHE for H{horizon}. Skipping.")
             return None, None, None, None, None, None

        print(f"Data preparation for H{horizon} complete. Train shape: {X_train_final.shape}, Test shape: {X_test_final.shape}") # type: ignore

        return X_train_final, X_test_final, y_train, y_test, X_train_main_unscaled, X_test_main_unscaled # type: ignore
