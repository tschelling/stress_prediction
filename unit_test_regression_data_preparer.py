import unittest
import pandas as pd
import numpy as np
import regression_data_preparer
import importlib
importlib.reload(regression_data_preparer)  # Ensure the latest version of PanelDataPreparer is used
from regression_data_preparer import RegressionDataPreparer

class TestRegressionDataPreparer(unittest.TestCase):

    def setUp(self):
        """Set up sample data and config for tests."""
        ids = ['bank_A', 'bank_B', 'bank_C', 'bank_D']
        dates = pd.to_datetime(pd.date_range('2020-01-01', periods=10, freq='Q'))
        multi_index = pd.MultiIndex.from_product([ids, dates], names=['id', 'date'])

        data = {
            'target': np.random.rand(len(multi_index)),
            'feature1': np.random.rand(len(multi_index)),
            'feature2': np.random.randn(len(multi_index)) * 100, # For outlier testing
            'total_assets': np.random.uniform(1e5, 1e7, len(multi_index)),
            'deposit_ratio': np.random.uniform(0.1, 0.9, len(multi_index)),
        }
        self.sample_df = pd.DataFrame(data, index=multi_index)

        # Introduce NaNs strategically
        # Bank A: no NaNs
        # Bank B: <30% NaNs in feature1
        self.sample_df.loc[('bank_B', dates[0]), 'feature1'] = np.nan
        self.sample_df.loc[('bank_B', dates[1]), 'feature1'] = np.nan
        # Bank C: >30% NaNs across its rows (e.g. 4 out of 10 obs for feature1, and 1 for target)
        # Total cells for bank C = 10 rows * 5 columns = 50
        # 30% of 50 = 15. Let's make 16 NaNs for bank_C
        for i in range(4): # 4 NaNs in feature1
            self.sample_df.loc[('bank_C', dates[i]), 'feature1'] = np.nan
        for i in range(4): # 4 NaNs in feature2
            self.sample_df.loc[('bank_C', dates[i+2]), 'feature2'] = np.nan
        for i in range(4): # 4 NaNs in total_assets
            self.sample_df.loc[('bank_C', dates[i+4]), 'total_assets'] = np.nan
        for i in range(4): # 4 NaNs in deposit_ratio
            self.sample_df.loc[('bank_C', dates[i+6]), 'deposit_ratio'] = np.nan

        # Bank D: some NaNs in target for imputation test
        self.sample_df.loc[('bank_D', dates[3]), 'target'] = np.nan
        self.sample_df.loc[('bank_D', dates[4]), 'target'] = np.nan

        self.base_config = {
            'TARGET_VARIABLE': 'target',
            'FEATURE_VARIABLES': ['feature1', 'feature2', 'total_assets', 'deposit_ratio'],
            'INCLUDE_TIME_FE': False,
            'INCLUDE_BANK_FE': False,
            'OUTLIER_THRESHOLD_TARGET': 3.0,
            'MISSING_VALUE_BANK_REMOVAL_THRESHOLD': 0.3,
            'DATA_BEGIN': None,
            'DATA_END': None,
            'RESTRICT_TO_NUMBER_OF_BANKS': None,
            'RESTRICT_TO_BANK_SIZE': None,
            'RESTRICT_TO_MINIMAL_DEPOSIT_RATIO': None,
            'RESTRICT_TO_MAX_CHANGE_IN_DEPOSIT_RATIO': None,
            'INCLUDE_AUTOREGRESSIVE_LAGS': False,
            'NUMBER_OF_LAGS_TO_INCLUDE': 0,
            'TRAIN_TEST_SPLIT_DIMENSION': 'date',
            'TEST_SPLIT': 0.2,
            'WINSORIZE_DO': False,
            'WINSORIZE_COLS': [],
        }

    def test_analyze_and_impute_missing_values(self):
        preparer = RegressionDataPreparer(self.sample_df.copy(), self.base_config)
        df_input = self.sample_df.copy()
        threshold = 0.3 # Same as in base_config for this test

        # Manually calculate expected initial stats
        initial_rows = len(df_input)
        initial_rows_with_mv = df_input.isnull().any(axis=1).sum()
        initial_banks_with_mv = df_input.groupby(level='id').apply(lambda x: x.isnull().any().any()).sum()

        df_processed, stats = preparer._process_missing_values(df_input, threshold)

        # 1. Check initial stats
        self.assertEqual(stats['initial']['num_rows'], initial_rows)
        self.assertEqual(stats['initial']['num_rows_with_mv'], initial_rows_with_mv)
        self.assertEqual(stats['initial']['num_banks_with_any_missing'], initial_banks_with_mv)

        # 2. Check bank removal
        # Bank C should be removed (has 16 NaNs / 50 total cells = 32% > 30%)
        self.assertNotIn('bank_C', df_processed.index.get_level_values('id').unique())
        self.assertIn('bank_A', df_processed.index.get_level_values('id').unique())
        self.assertIn('bank_B', df_processed.index.get_level_values('id').unique())
        self.assertIn('bank_D', df_processed.index.get_level_values('id').unique())
        self.assertEqual(stats['after_bank_removal']['num_banks_removed'], 1)

        # 3. Check imputation
        self.assertEqual(df_processed.isnull().sum().sum(), 0, "DataFrame should have no NaNs after imputation")
        self.assertEqual(stats['after_imputation']['num_rows_with_mv'], 0)
        self.assertEqual(stats['after_imputation']['num_banks_with_any_missing'], 0)

        # Check specific imputed value for bank_D
        # NaNs were at dates[3] and dates[4] for bank_D target
        # ffill should fill dates[3] with value from dates[2]
        # bfill should fill dates[4] with value from dates[5] (if ffill didn't get it)
        # Since it's ffill().bfill(), dates[3] gets dates[2]'s val, then dates[4] gets dates[3]'s val (which is now dates[2]'s val)
        original_bank_d_target_val_at_2 = self.sample_df.loc[('bank_D', self.sample_df.index.get_level_values('date')[2]), 'target']
        imputed_bank_d_target_val_at_3 = df_processed.loc[('bank_D', self.sample_df.index.get_level_values('date')[3]), 'target']
        imputed_bank_d_target_val_at_4 = df_processed.loc[('bank_D', self.sample_df.index.get_level_values('date')[4]), 'target']
        
        self.assertEqual(imputed_bank_d_target_val_at_3, original_bank_d_target_val_at_2)
        self.assertEqual(imputed_bank_d_target_val_at_4, original_bank_d_target_val_at_2) # Due to ffill then bfill sequence

    def test_winsorize_data(self):
        config = self.base_config.copy()
        config['WINSORIZE_DO'] = True
        config['WINSORIZE_COLS'] = ['feature1', 'feature2']
        config['WINSORIZE_LOWER_PERCENTILE'] = 0.05
        config['WINSORIZE_UPPER_PERCENTILE'] = 0.95

        # Create a preparer instance to set winsorization attributes
        preparer = RegressionDataPreparer(self.sample_df.copy(), config)
        
        # Call _winsorize_data directly for focused testing
        # Need to handle NaNs before winsorization for a clean test of winsorization logic
        df_no_nan = self.sample_df.copy().fillna(0) # Simple fill for test
        winsorized_df = preparer._winsorize_data(df_no_nan)

        for col in ['feature1', 'feature2']:
            original_col = df_no_nan[col]
            winsorized_col = winsorized_df[col]
            lower_bound = original_col.quantile(0.05)
            upper_bound = original_col.quantile(0.95)
            self.assertTrue((winsorized_col >= lower_bound).all())
            self.assertTrue((winsorized_col <= upper_bound).all())
            # Check if values outside bounds were clipped
            self.assertTrue(winsorized_col[original_col < lower_bound].eq(lower_bound).all())
            self.assertTrue(winsorized_col[original_col > upper_bound].eq(upper_bound).all())

    def test_remove_outliers(self):
        df_test = self.sample_df.copy()
        # Introduce a clear outlier in 'target' for bank_A
        df_test.loc[('bank_A', df_test.index.get_level_values('date')[0]), 'target'] = 1000 
        
        preparer = RegressionDataPreparer(df_test, self.base_config) # Config has threshold 3.0
        
        # Call _remove_outliers directly
        processed_df = preparer._remove_outliers(df_test, 'target', threshold=3.0)
        
        self.assertTrue(len(processed_df) < len(df_test))
        # Check if the specific outlier row was removed
        self.assertNotIn(('bank_A', df_test.index.get_level_values('date')[0]), processed_df.index)

    def test_restrict_sample_logic(self):
        config = self.base_config.copy()
        config['DATA_BEGIN'] = '2020-04-01' # Second quarter
        config['RESTRICT_TO_NUMBER_OF_BANKS'] = 2
        
        preparer = RegressionDataPreparer(self.sample_df.copy(), config)
        # _restrict_sample_logic is called within _prepare_initial_data
        # We can inspect preparer.base_data_for_horizons
        
        processed_df = preparer.final_data
        
        # Check date restriction
        self.assertTrue((processed_df.index.get_level_values('date') >= pd.to_datetime('2020-04-01')).all())
        
        # Check number of banks (after other processing steps in _prepare_initial_data)
        # Note: The number of banks might be further reduced by missing value removal, outlier removal etc.
        # So, we check if it's AT MOST the restricted number.
        # Bank C is removed by missing value step, leaving A, B, D. Then 2 are sampled.
        self.assertLessEqual(processed_df.index.get_level_values('id').nunique(), 2)

    def test_get_horizon_specific_data_runs_without_error(self):
        config = self.base_config.copy()
        config['INCLUDE_AUTOREGRESSIVE_LAGS'] = True
        config['NUMBER_OF_LAGS_TO_INCLUDE'] = 1
        config['INCLUDE_TIME_FE'] = True

        preparer = RegressionDataPreparer(self.sample_df.copy(), config)
        
        # Bank C is removed due to missing values.
        # Bank A, B, D remain.
        # Outlier removal might remove some rows.
        # Date split might make train/test small.

        X_train, X_test, y_train, y_test, _, _ = preparer.get_horizon_specific_data(horizon=1)

        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        # Test set can be None or empty if split results in no test data after all processing
        # So we don't assert X_test is not None strictly.

        if X_train is not None and not X_train.empty:
            self.assertTrue(any(col.startswith('num__target_ar_lag_1') for col in X_train.columns))
            self.assertTrue(any(col.startswith('cat__quarter_') for col in X_train.columns)) # Check for OHE time FE
            self.assertEqual(len(X_train), len(y_train))

        if X_test is not None and not X_test.empty:
            self.assertTrue(any(col.startswith('num__target_ar_lag_1') for col in X_test.columns))
            self.assertTrue(any(col.startswith('cat__quarter_') for col in X_test.columns)) # Check for OHE time FE
            self.assertEqual(len(X_test), len(y_test))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)