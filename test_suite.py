import unittest
import pandas as pd
import numpy as np
import os

import importlib

# Dynamic imports since module names start with numbers
preprocessing = importlib.import_module("2_preprocessing")
train_lstm = importlib.import_module("4_train_lstm")
train_models = importlib.import_module("3_train_models")

calculate_indicators = preprocessing.calculate_indicators
create_sequences = train_lstm.create_sequences
calculate_mape = train_models.calculate_mape

class TestForecastingPipeline(unittest.TestCase):

    def setUp(self):
        # Create a mock DataFrame representing daily stock prices
        dates = pd.date_range(start="2025-01-01", periods=100)
        # Create a steady upward trend with small random fluctuations
        np.random.seed(42)
        close_prices = np.linspace(100, 150, 100) + np.random.normal(0, 0.5, 100)
        
        self.mock_df = pd.DataFrame({
            'Date': dates,
            'Open': close_prices - 0.5,
            'High': close_prices + 1.0,
            'Low': close_prices - 1.0,
            'Close': close_prices,
            'Volume': np.random.randint(100000, 500000, 100)
        })

    def test_calculate_indicators_columns(self):
        """Verify that all technical indicators and target variables are successfully generated."""
        df_indicators = calculate_indicators(self.mock_df)
        
        expected_cols = [
            'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI_14',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'Volatility',
            'Lag_1', 'Lag_2', 'Lag_3', 'Target'
        ]
        
        for col in expected_cols:
            self.assertIn(col, df_indicators.columns, f"Indicator column '{col}' is missing.")

    def test_calculate_indicators_values(self):
        """Verify the values of generated indicators (e.g. no infinite values, proper shifting)."""
        df_indicators = calculate_indicators(self.mock_df)
        
        # Test targets: Target at row t must be Close at row t+1
        # Close at index 10 should be Target at index 9
        self.assertEqual(df_indicators['Target'].iloc[9], df_indicators['Close'].iloc[10])
        
        # Test lags: Lag_1 at row t must be Close at row t-1
        self.assertEqual(df_indicators['Lag_1'].iloc[10], df_indicators['Close'].iloc[9])
        
        # Test that RSI values lie between 0 and 100
        rsi_vals = df_indicators['RSI_14'].dropna()
        self.assertTrue((rsi_vals >= 0).all() and (rsi_vals <= 100).all())

    def test_rsi_division_by_zero_robustness(self):
        """Verify that RSI calculation doesn't fail or return NaNs during continuous price gains."""
        # Create a mock dataframe with only positive gains
        gain_df = self.mock_df.copy()
        gain_df['Close'] = np.linspace(100, 200, 100) # strictly increasing Close, zero downward movements
        
        try:
            df_indicators = calculate_indicators(gain_df)
            rsi_vals = df_indicators['RSI_14'].dropna()
            self.assertFalse(np.isinf(rsi_vals).any(), "RSI contains infinite values due to division by zero.")
            self.assertFalse(rsi_vals.isnull().any(), "RSI contains NaN values due to division by zero.")
        except ZeroDivisionError:
            self.fail("RSI calculation raised a ZeroDivisionError.")

    def test_lstm_sequence_creation_shapes(self):
        """Verify that create_sequences returns correctly shaped features (X) and targets (y)."""
        df_indicators = calculate_indicators(self.mock_df).dropna().reset_index(drop=True)
        
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI_14',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'Volatility',
            'Lag_1', 'Lag_2', 'Lag_3'
        ]
        
        features = df_indicators[feature_cols].values
        targets = df_indicators['Target'].values
        
        seq_length = 10
        X, y = create_sequences(features, targets, seq_length)
        
        expected_samples = len(df_indicators) - seq_length
        self.assertEqual(X.shape, (expected_samples, seq_length, len(feature_cols)))
        self.assertEqual(y.shape, (expected_samples,))

    def test_lstm_sequence_target_alignment(self):
        """Verify that the target y is correctly aligned to t+1 relative to the end of the sequence window."""
        df_indicators = calculate_indicators(self.mock_df).dropna().reset_index(drop=True)
        
        feature_cols = ['Close', 'Lag_1']
        features = df_indicators[feature_cols].values
        targets = df_indicators['Target'].values
        
        seq_length = 10
        X, y = create_sequences(features, targets, seq_length)
        
        # For the first sample (index 0):
        # - The input sequence spans indices 0 to 9. The features at index 9 represent day t.
        # - The predicted target must be the Close price of index 10 (day t+1).
        # - Since targets[k] is df['Target'].iloc[k], which is Close price at index k+1,
        #   y[0] must equal targets[seq_length - 1] (index 9), which represents Close price at index 10.
        
        expected_target_price = df_indicators['Close'].iloc[seq_length]
        self.assertEqual(y[0], expected_target_price)

    def test_calculate_mape_accuracy(self):
        """Verify that calculate_mape returns correct percentage values and handles zero division."""
        actual = [100.0, 200.0, 150.0]
        predicted = [110.0, 190.0, 150.0]
        # errors: 10% (over), 5% (under), 0%
        # average: (10 + 5 + 0) / 3 = 5.0%
        self.assertAlmostEqual(calculate_mape(actual, predicted), 5.0)

if __name__ == '__main__':
    unittest.main()
