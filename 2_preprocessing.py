import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def calculate_indicators(df):
    """
    Calculates technical indicators: SMA(10,50), EMA(10,50), RSI(14), MACD,
    Bollinger Bands, Volatility, and Lag features.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 2. Relative Strength Index (RSI 14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4. Bollinger Bands (20-day window, 2 standard deviations)
    bb_mid = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Middle'] = bb_mid
    df['BB_Upper'] = bb_mid + (2 * bb_std)
    df['BB_Lower'] = bb_mid - (2 * bb_std)
    
    # 5. Volatility (20-day rolling standard deviation of daily returns)
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    # 6. Lag features (t-1, t-2, t-3 Close prices)
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    
    # Target variable: next day's close price (t+1)
    df['Target'] = df['Close'].shift(-1)
    
    return df

def preprocess_ticker(ticker, raw_dir, processed_dir, models_dir):
    """
    Cleans data, adds indicators, splits chronologically, and normalizes/scales data.
    """
    raw_path = os.path.join(raw_dir, f"{ticker}.csv")
    if not os.path.exists(raw_path):
        print(f"Error: Raw file {raw_path} not found.")
        return
        
    df = pd.read_csv(raw_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Drop zero-volume days (halted trading)
    initial_len = len(df)
    df = df[df['Volume'] > 0].reset_index(drop=True)
    dropped_vol = initial_len - len(df)
    
    # Forward-fill small gaps, drop rows still incomplete
    df = df.ffill().dropna().reset_index(drop=True)
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Drop rows with NaN values created by indicators/lags/target shifting
    df = df.dropna().reset_index(drop=True)
    
    # Define features and target
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI_14',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'Volatility',
        'Lag_1', 'Lag_2', 'Lag_3'
    ]
    target_col = 'Target'
    
    # Chronological Split (80% Train, 20% Test)
    # CRITICAL: We do NOT use random split to prevent look-ahead bias in time series.
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Save unscaled datasets for baseline references, backtesting, and recommendations
    train_df.to_csv(os.path.join(processed_dir, f"{ticker}_train_unscaled.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, f"{ticker}_test_unscaled.csv"), index=False)
    
    # CRITICAL: Scaling/normalization is fitted ONLY on the training set to prevent data leakage.
    # We fit separate scalers for features and target.
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit and transform training features
    train_x_scaled = scaler_x.fit_transform(train_df[feature_cols])
    # Transform test features
    test_x_scaled = scaler_x.transform(test_df[feature_cols])
    
    # Fit and transform training target
    train_y_scaled = scaler_y.fit_transform(train_df[[target_col]])
    # Transform test target
    test_y_scaled = scaler_y.transform(test_df[[target_col]])
    
    # Save the scaler models for deployment/inference
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler_x, os.path.join(models_dir, f"{ticker}_scaler_x.joblib"))
    joblib.dump(scaler_y, os.path.join(models_dir, f"{ticker}_scaler_y.joblib"))
    
    # Reconstruct scaled dataframes with original metadata (Date, Ticker)
    train_scaled_df = pd.DataFrame(train_x_scaled, columns=feature_cols)
    train_scaled_df['Target'] = train_y_scaled
    train_scaled_df.insert(0, 'Date', train_df['Date'].values)
    
    test_scaled_df = pd.DataFrame(test_x_scaled, columns=feature_cols)
    test_scaled_df['Target'] = test_y_scaled
    test_scaled_df.insert(0, 'Date', test_df['Date'].values)
    
    train_scaled_df.to_csv(os.path.join(processed_dir, f"{ticker}_train_scaled.csv"), index=False)
    test_scaled_df.to_csv(os.path.join(processed_dir, f"{ticker}_test_scaled.csv"), index=False)
    
    print(f"Processed {ticker}:")
    print(f"  - Dropped {dropped_vol} zero-volume rows.")
    print(f"  - Chronological split: {len(train_df)} train rows, {len(test_df)} test rows.")
    print(f"  - Saved scaled and unscaled datasets to {processed_dir}")
    print(f"  - Saved scalers to {models_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess stock market data.")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help="List of stock tickers to preprocess.")
    parser.add_argument('--raw_dir', default='data/raw', help="Directory containing raw data")
    parser.add_argument('--processed_dir', default='data/processed', help="Directory to save processed data")
    parser.add_argument('--models_dir', default='models', help="Directory to save scalers")
    
    args = parser.parse_args()
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    
    for ticker in args.tickers:
        preprocess_ticker(ticker, args.raw_dir, args.processed_dir, args.models_dir)
