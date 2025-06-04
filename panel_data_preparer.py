import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from typing import Union, Tuple, Optional, List, Dict, Any

class PanelDataPreparer:
    def __init__(self, initial_df: pd.DataFrame, config: Dict[str, Any]):
        self.initial_df = initial_df.copy()
        self.config = config.copy()

        self.target_variable: str = self.config['TARGET_VARIABLE']
        self.feature_variables_base: List[str] = self.config['FEATURE_VARIABLES'][:]
        
        self.include_time_fe: bool = self.config.get('INCLUDE_TIME_FE', False)
        self.include_bank_fe: bool = self.config.get('INCLUDE_BANK_FE', False)
        self.include_ar_lags: bool = self.config.get('INCLUDE_AUTOREGRESSIVE_LAGS', True)
        self.num_lags_to_include: int = self.config.get('NUMBER_OF_LAGS_TO_INCLUDE', 0)

        self.train_test_split_dimension: str = self.config['TRAIN_TEST_SPLIT_DIMENSION']
        self.test_split_config: Union[str, float] = self.config['TEST_SPLIT']

        self.base_data_for_horizons: Optional[pd.DataFrame] = None
        self.final_feature_list_for_horizon_processing: List[str] = self.feature_variables_base[:]

        self._prepare_initial_data()

    def _remove_outliers(self, df: pd.DataFrame, target_col: str, threshold: float) -> pd.DataFrame:
        print(f"Removing outliers from the target variable '{target_col}' using z-score threshold {threshold}...")
        nr_rows_before = df.shape[0]
        z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
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
            df_processed['abs_change_deposit_ratio'] = df_processed.groupby(level='id')['deposit_ratio'].diff().abs().fillna(0)
            df_processed = df_processed[df_processed['abs_change_deposit_ratio'] <= max_change_deposit_ratio]
            df_processed = df_processed.drop(columns=['abs_change_deposit_ratio'])
            print(f"Filtered by max change in deposit ratio (<= {max_change_deposit_ratio}). Rows remaining: {len(df_processed)}")

        if bank_size_threshold is not None:
            if 'total_assets' not in df_processed.columns:
                print("Warning: 'total_assets' column not found. Cannot filter by bank_size_threshold.")
            else:
                avg_assets = df_processed.groupby(level='id')['total_assets'].mean()
                ids_meeting_size_criteria = avg_assets[avg_assets >= bank_size_threshold].index
                df_processed = df_processed[df_processed.index.get_level_values('id').isin(ids_meeting_size_criteria)]
                print(f"Filtered to banks with average total assets >= {bank_size_threshold}. Banks: {df_processed.index.get_level_values('id').nunique()}, Rows: {len(df_processed)}")

        if deposit_ratio_threshold is not None:
            if 'deposit_ratio' not in df_processed.columns:
                print("Warning: 'deposit_ratio' column not found. Cannot filter by deposit_ratio_threshold.")
            else:
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
        df = self.initial_df
        print(f"\n--- Preparing Initial Data ---")
        print(f"Initial raw data shape: {df.shape}")

        current_feature_list = self.feature_variables_base[:]

        if self.include_time_fe:
            df['quarter'] = 'quarter_' + df.index.get_level_values('date').quarter.astype(str)
            df['year'] = 'year_' + df.index.get_level_values('date').year.astype(str)
            if 'quarter' not in current_feature_list: current_feature_list.append('quarter')
            if 'year' not in current_feature_list: current_feature_list.append('year')
            print("Added time fixed effects: 'quarter', 'year'.")

        if self.include_bank_fe:
            df['bank_id'] = 'bank_id_' + df.index.get_level_values('id').astype('category').codes.astype(str)
            if 'bank_id' not in current_feature_list:
                current_feature_list = ['bank_id'] + current_feature_list # Prepend
            print("Added bank fixed effect: 'bank_id'.")
        
        self.final_feature_list_for_horizon_processing = current_feature_list

        cols_for_dropna = self.final_feature_list_for_horizon_processing + [self.target_variable]
        # Ensure all columns for dropna actually exist in df
        cols_for_dropna = [col for col in cols_for_dropna if col in df.columns]
        
        print(f"Dropping NaNs based on subset: {cols_for_dropna}")
        rows_before_na_drop = len(df)
        df = df.dropna(subset=cols_for_dropna).copy()
        print(f"Rows before NaN drop: {rows_before_na_drop}, after: {len(df)}. Removed: {rows_before_na_drop - len(df)}")

        df = self._remove_outliers(df, self.target_variable, threshold=self.config.get('OUTLIER_THRESHOLD_TARGET', 3.0))
        df = self._remove_banks_with_few_observations(df, min_observations=self.config.get('MIN_OBS_PER_BANK', 10))
        
        df = self._restrict_sample_logic(df)
        
        self.base_data_for_horizons = df.copy()
        print(f"Base data prepared. Shape: {self.base_data_for_horizons.shape}")
        print(f"Final features for horizon processing: {self.final_feature_list_for_horizon_processing}")

    def get_horizon_specific_data(self, horizon: int) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series],
        Optional[pd.DataFrame], Optional[pd.DataFrame]
    ]:
        if self.base_data_for_horizons is None or self.base_data_for_horizons.empty:
            print(f"Base data is not prepared or is empty. Cannot generate data for horizon {horizon}.")
            return None, None, None, None, None, None

        df_h = self.base_data_for_horizons.copy()
        df_h = df_h.sort_index()

        ar_term_names = []
        if self.include_ar_lags and self.num_lags_to_include > 0:
            for lag_num in range(1, self.num_lags_to_include + 1):
                ar_lag_col_name = f'{self.target_variable}_ar_lag_{lag_num}'
                df_h[ar_lag_col_name] = df_h.groupby(level='id', group_keys=False)[self.target_variable].shift(lag_num)
                ar_term_names.append(ar_lag_col_name)

        shifted_target_col = f'{self.target_variable}_target_h{horizon}'
        df_h[shifted_target_col] = df_h.groupby(level='id', group_keys=False)[self.target_variable].shift(-horizon)

        cols_to_check_for_na = [shifted_target_col] + ar_term_names
        df_h.dropna(subset=cols_to_check_for_na, inplace=True)
        
        ar_term_names = [name for name in ar_term_names if name in df_h.columns] # Re-validate after dropna

        if df_h.empty:
            print(f"DataFrame empty for H{horizon} after target/AR shift & NaN drop. Skipping.")
            return None, None, None, None, None, None
        
        y_full = df_h[shifted_target_col]
        
        # Features for Prophet are the original ones (before lagging main features)
        # self.final_feature_list_for_horizon_processing includes bank_id, quarter, year if configured
        prophet_feature_list = [f for f in self.final_feature_list_for_horizon_processing if f in df_h.columns]
        X_pre_lag_full = df_h[prophet_feature_list].copy()

        if X_pre_lag_full.empty or y_full.empty:
            print(f"X_pre_lag_full or y_full empty for H{horizon} before split. Skipping.")
            return None, None, None, None, None, None

        # --- Train/Test Split ---
        X_train_pre_lag, X_test_pre_lag, y_train, y_test = None, None, None, None
        # (Split logic from test3.py's prepare_data_for_horizon, adapted to use self.config)
        if self.train_test_split_dimension == "date":
            unique_dates = X_pre_lag_full.index.get_level_values('date').unique().sort_values()
            if len(unique_dates) < 2: return None, None, None, None, None, None # Not enough dates
            split_date_val: Optional[pd.Timestamp] = None
            if isinstance(self.test_split_config, str):
                try: split_date_val = pd.to_datetime(self.test_split_config)
                except ValueError: return None, None, None, None, None, None # Invalid date string
                if split_date_val < unique_dates.min() or split_date_val >= unique_dates.max(): return None, None, None, None, None, None
            elif isinstance(self.test_split_config, float) and 0.0 < self.test_split_config < 1.0:
                num_test_dates = int(np.ceil(len(unique_dates) * self.test_split_config))
                if num_test_dates == 0: num_test_dates = 1
                split_idx = len(unique_dates) - num_test_dates
                if split_idx <= 0 or split_idx >= len(unique_dates): return None, None, None, None, None, None
                split_date_val = unique_dates[split_idx -1]
            else: return None, None, None, None, None, None # Invalid split config
            
            train_mask = X_pre_lag_full.index.get_level_values('date') <= split_date_val
            test_mask = X_pre_lag_full.index.get_level_values('date') > split_date_val
        elif self.train_test_split_dimension == "id":
            if not (isinstance(self.test_split_config, float) and 0.0 < self.test_split_config < 1.0): return None, None, None, None, None, None
            unique_ids = X_pre_lag_full.index.get_level_values('id').unique()
            if len(unique_ids) < 2: return None, None, None, None, None, None
            num_test_ids = int(np.ceil(len(unique_ids) * self.test_split_config))
            if self.test_split_config > 0 and num_test_ids == 0: num_test_ids = 1
            if num_test_ids == 0 or num_test_ids >= len(unique_ids): return None, None, None, None, None, None
            
            gss = GroupShuffleSplit(n_splits=1, test_size=num_test_ids, random_state=42)
            # Need to reset index for GSS groups argument
            temp_X_reset = X_pre_lag_full.reset_index()
            temp_y_reset = y_full.reset_index() # Not used for splitting y, but for consistency
            try:
                train_indices, test_indices = next(gss.split(temp_X_reset, temp_y_reset, groups=temp_X_reset['id']))
            except ValueError: return None, None, None, None, None, None # Error in GSS
            
            # Create masks from indices
            train_mask = X_pre_lag_full.index.isin(temp_X_reset.iloc[train_indices].set_index(['id','date']).index)
            test_mask = X_pre_lag_full.index.isin(temp_X_reset.iloc[test_indices].set_index(['id','date']).index)
        else:
            raise ValueError("train_test_split_dimension must be 'date' or 'id'")

        X_train_pre_lag = X_pre_lag_full[train_mask]
        X_test_pre_lag = X_pre_lag_full[test_mask]
        y_train = y_full[train_mask]
        y_test = y_full[test_mask]

        # --- Feature Lags for main models (applied after initial split) ---
        # Identify numeric features from prophet_feature_list (which are the original features)
        numeric_features_for_lags = [
            f for f in prophet_feature_list
            if f in X_train_pre_lag.columns and pd.api.types.is_numeric_dtype(X_train_pre_lag[f]) and f != self.target_variable
        ]
        # Categorical features are those in prophet_feature_list not in numeric_features_for_lags
        categorical_model_features = [
            f for f in prophet_feature_list
            if f in X_train_pre_lag.columns and f not in numeric_features_for_lags and f != self.target_variable
        ]

        def _add_lags_to_df_internal(df_to_lag: pd.DataFrame, num_features_to_lag: List[str], num_lags: int) -> Tuple[pd.DataFrame, List[str]]:
            df_out = df_to_lag.copy()
            lagged_names: List[str] = []
            if num_lags > 0:
                for feat_name in num_features_to_lag:
                    for lag_i in range(1, num_lags + 1):
                        lag_col = f"{feat_name}_lag_{lag_i}"
                        df_out[lag_col] = df_out.groupby(level='id', group_keys=False)[feat_name].shift(lag_i)
                        lagged_names.append(lag_col)
            return df_out, lagged_names

        X_train_with_feature_lags, train_lagged_names = _add_lags_to_df_internal(
            X_train_pre_lag, numeric_features_for_lags, self.num_lags_to_include
        )
        
        X_test_with_feature_lags = pd.DataFrame(index=X_test_pre_lag.index) # Init empty
        if not X_test_pre_lag.empty:
            # For test lags, concat train_pre_lag and test_pre_lag to avoid lookahead bias within test but allow history from train
            combined_for_test_feature_lags = pd.concat([X_train_pre_lag, X_test_pre_lag]).sort_index()
            combined_with_feature_lags, _ = _add_lags_to_df_internal(
                combined_for_test_feature_lags, numeric_features_for_lags, self.num_lags_to_include
            )
            X_test_with_feature_lags = combined_with_feature_lags.loc[X_test_pre_lag.index]

        # Define final features for main models
        final_model_feature_names = categorical_model_features[:]
        if self.num_lags_to_include > 0:
            final_model_feature_names.extend(train_lagged_names)
        else: # No feature lags, use original numeric features
            final_model_feature_names.extend(numeric_features_for_lags)
        
        # Add AR terms (if any) to this list
        final_model_feature_names.extend(ar_term_names)
        final_model_feature_names = sorted(list(set(final_model_feature_names))) # Unique and sorted

        # Select these features for X_train_main_unscaled / X_test_main_unscaled
        # X_train_with_feature_lags already has original cat/num features + their lags. Need to add AR terms.
        X_train_main_unscaled = X_train_with_feature_lags.copy()
        X_test_main_unscaled = X_test_with_feature_lags.copy()

        for ar_name in ar_term_names: # Add AR terms from df_h
            X_train_main_unscaled[ar_name] = df_h.loc[X_train_main_unscaled.index, ar_name]
            if not X_test_main_unscaled.empty:
                 X_test_main_unscaled[ar_name] = df_h.loc[X_test_main_unscaled.index, ar_name]
        
        # Ensure only final_model_feature_names are present and in order
        X_train_main_unscaled = X_train_main_unscaled[final_model_feature_names]
        if not X_test_main_unscaled.empty:
            X_test_main_unscaled = X_test_main_unscaled[final_model_feature_names]
        else: # Ensure empty test df has correct columns
            X_test_main_unscaled = pd.DataFrame(columns=final_model_feature_names, index=X_test_main_unscaled.index)

        # Drop NaNs from lagging features and AR terms, then align y
        X_train_main_unscaled.dropna(inplace=True)
        y_train = y_train.loc[X_train_main_unscaled.index]

        if not X_test_main_unscaled.empty:
            X_test_main_unscaled.dropna(inplace=True)
            y_test = y_test.loc[X_test_main_unscaled.index]

        # Data for Prophet (original features, aligned with final y_train/y_test)
        X_train_prophet_unscaled = X_train_pre_lag[prophet_feature_list].loc[y_train.index] if not y_train.empty else pd.DataFrame(columns=prophet_feature_list)
        X_test_prophet_unscaled = X_test_pre_lag[prophet_feature_list].loc[y_test.index] if not y_test.empty else pd.DataFrame(columns=prophet_feature_list)

        if X_train_main_unscaled.empty or y_train.empty:
            print(f"Train set empty after final processing for H{horizon}. Skipping.")
            return None, None, None, None, None, None

        # --- Scaling and OHE ---
        numeric_cols_for_scaling = X_train_main_unscaled.select_dtypes(include=np.number).columns.tolist()
        categorical_cols_for_ohe = X_train_main_unscaled.select_dtypes(include=['object', 'category']).columns.tolist()

        transformers_list = []
        if numeric_cols_for_scaling:
            transformers_list.append(('num', StandardScaler(), numeric_cols_for_scaling))
        if categorical_cols_for_ohe:
            transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None), categorical_cols_for_ohe))

        if not transformers_list: # No numeric or categorical features to transform
            X_train_final = X_train_main_unscaled.copy()
            X_test_final = X_test_main_unscaled.copy()
        else:
            preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')
            
            X_train_transformed_np = preprocessor.fit_transform(X_train_main_unscaled)
            transformed_names = preprocessor.get_feature_names_out()
            X_train_final = pd.DataFrame(X_train_transformed_np, columns=transformed_names, index=X_train_main_unscaled.index)

            if not X_test_main_unscaled.empty:
                X_test_transformed_np = preprocessor.transform(X_test_main_unscaled)
                X_test_final = pd.DataFrame(X_test_transformed_np, columns=transformed_names, index=X_test_main_unscaled.index)
            else:
                X_test_final = pd.DataFrame(columns=transformed_names, index=X_test_main_unscaled.index)
        
        return X_train_final, X_test_final, y_train, y_test, X_train_prophet_unscaled, X_test_prophet_unscaled