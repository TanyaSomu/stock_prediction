import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def calculate_mape(actual, predicted):
    """
    Calculates Mean Absolute Percentage Error (MAPE) safely.
    """
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def create_sequences(features, targets, seq_length=60):
    """
    Constructs sequence windows for LSTM training.
    past seq_length days -> predict next day
    """
    X = []
    y = []
    for i in range(len(features) - seq_length):
        X.append(features[i:(i + seq_length)])
        y.append(targets[i + seq_length - 1])
    return np.array(X), np.array(y)

def train_and_evaluate_lstm(ticker, processed_dir, models_dir, outputs_dir, epochs=10, batch_size=32):
    """
    Trains and evaluates LSTM model with and without sentiment.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load target scaler for inverse transforming
    scaler_y_path = os.path.join(models_dir, f"{ticker}_scaler_y.joblib")
    if not os.path.exists(scaler_y_path):
        print(f"Error: Target scaler not found for {ticker} at {scaler_y_path}")
        return
    scaler_y = joblib.load(scaler_y_path)
    
    configs = [
        ("", "Without Sentiment"),
        ("_sentiment", "With Sentiment")
    ]
    
    metrics_records = []
    seq_length = 60
    
    for suffix, feat_label in configs:
        train_path = os.path.join(processed_dir, f"{ticker}_train_scaled{suffix}.csv")
        test_path = os.path.join(processed_dir, f"{ticker}_test_scaled{suffix}.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Data files {train_path} or {test_path} missing. Skipping LSTM '{feat_label}'.")
            continue
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Date column
        test_dates = test_df['Date'].values[seq_length - 1 : -1]
        
        # Prepare inputs and targets
        # Drop metadata columns
        train_feats = train_df.drop(columns=['Date', 'Target']).values
        train_targets = train_df['Target'].values
        
        test_feats = test_df.drop(columns=['Date', 'Target']).values
        test_targets = test_df['Target'].values
        
        # Create sequences
        X_train, y_train = create_sequences(train_feats, train_targets, seq_length)
        X_test, y_test = create_sequences(test_feats, test_targets, seq_length)
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Data length is too short to form {seq_length}-day sequences. Skipping LSTM.")
            continue
            
        # Inverse transform actual targets to get raw dollar prices
        y_test_raw = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Build LSTM Architecture
        num_features = X_train.shape[2]
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        print(f"Training LSTM ({feat_label}) for {ticker} with shape {X_train.shape}...")
        
        # Fit model
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Save model
        model_save_path = os.path.join(models_dir, f"{ticker}_lstm{suffix}.h5")
        model.save(model_save_path)
        
        # Predict and inverse scale
        preds_scaled = model.predict(X_test)
        preds_raw = scaler_y.inverse_transform(preds_scaled).flatten()
        
        # Calculate metrics in raw dollar space
        rmse = np.sqrt(mean_squared_error(y_test_raw, preds_raw))
        mae = mean_absolute_error(y_test_raw, preds_raw)
        mape = calculate_mape(y_test_raw, preds_raw)
        r2 = r2_score(y_test_raw, preds_raw)
        
        print(f"  LSTM ({feat_label}) Results -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")
        
        metrics_records.append({
            'Ticker': ticker,
            'Model': 'LSTM',
            'Features': feat_label,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })
        
        # Save actual vs predicted values for plotting and recommendation
        preds_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': y_test_raw,
            'Predicted': preds_raw
        })
        preds_save_path = os.path.join(models_dir, f"{ticker}_lstm{suffix}_predictions.csv")
        preds_df.to_csv(preds_save_path, index=False)
        
    # Append/Update outputs/metrics.csv
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = os.path.join(outputs_dir, "metrics.csv")
    
    if os.path.exists(metrics_csv_path):
        existing_metrics = pd.read_csv(metrics_csv_path)
        # Filter out existing LSTM records for this ticker to prevent duplication
        existing_metrics = existing_metrics[
            ~((existing_metrics['Ticker'] == ticker) & (existing_metrics['Model'] == 'LSTM'))
        ]
        combined_metrics = pd.concat([existing_metrics, metrics_df], ignore_index=True)
        combined_metrics.to_csv(metrics_csv_path, index=False)
    else:
        metrics_df.to_csv(metrics_csv_path, index=False)
        
    print(f"Finished LSTM training & validation for {ticker}. Saved metrics.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM deep learning model.")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help="Tickers to process.")
    parser.add_argument('--processed_dir', default='data/processed', help="Directory containing processed data")
    parser.add_argument('--models_dir', default='models', help="Directory to save models")
    parser.add_argument('--outputs_dir', default='outputs', help="Directory to save metrics")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()
    for ticker in args.tickers:
        train_and_evaluate_lstm(ticker, args.processed_dir, args.models_dir, args.outputs_dir, args.epochs, args.batch_size)
