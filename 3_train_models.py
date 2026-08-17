import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def calculate_mape(actual, predicted):
    """
    Calculates Mean Absolute Percentage Error (MAPE) safely.
    """
    actual, predicted = np.array(actual), np.array(predicted)
    # Avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def train_and_evaluate(ticker, processed_dir, models_dir, outputs_dir):
    """
    Trains and evaluates LR, RF, XGB, and SVR models.
    Runs for both baseline (no sentiment) and enhanced (with sentiment) configurations.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load target scaler for inverse transforming
    scaler_y_path = os.path.join(models_dir, f"{ticker}_scaler_y.joblib")
    if not os.path.exists(scaler_y_path):
        print(f"Error: Target scaler not found for {ticker} at {scaler_y_path}")
        return
    scaler_y = joblib.load(scaler_y_path)
    
    # Configurations to train: (suffix, name_label)
    configs = [
        ("", "Without Sentiment"),
        ("_sentiment", "With Sentiment")
    ]
    
    metrics_records = []
    
    for suffix, feat_label in configs:
        train_path = os.path.join(processed_dir, f"{ticker}_train_scaled{suffix}.csv")
        test_path = os.path.join(processed_dir, f"{ticker}_test_scaled{suffix}.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Data files {train_path} or {test_path} missing. Skipping configuration '{feat_label}'.")
            continue
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Separate features and target
        X_train = train_df.drop(columns=['Date', 'Target'])
        y_train = train_df['Target']
        X_test = test_df.drop(columns=['Date', 'Target'])
        y_test = test_df['Target']
        test_dates = test_df['Date']
        
        # Inverse transform actual target to get raw dollar prices
        y_test_raw = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        
        for name, model in models.items():
            print(f"Training {name} ({feat_label}) for {ticker}...")
            # Fit model
            model.fit(X_train, y_train)
            
            # Save the trained model
            model_save_path = os.path.join(models_dir, f"{ticker}_{name.lower().replace(' ', '_')}{suffix}.joblib")
            joblib.dump(model, model_save_path)
            
            # Predict and inverse scale
            preds_scaled = model.predict(X_test)
            preds_raw = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            
            # Calculate metrics in raw dollar space
            rmse = np.sqrt(mean_squared_error(y_test_raw, preds_raw))
            mae = mean_absolute_error(y_test_raw, preds_raw)
            mape = calculate_mape(y_test_raw, preds_raw)
            r2 = r2_score(y_test_raw, preds_raw)
            
            print(f"  {name} Results -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")
            
            metrics_records.append({
                'Ticker': ticker,
                'Model': name,
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
            preds_save_path = os.path.join(models_dir, f"{ticker}_{name.lower().replace(' ', '_')}{suffix}_predictions.csv")
            preds_df.to_csv(preds_save_path, index=False)
            
    # Save/Append metrics to outputs/metrics.csv
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = os.path.join(outputs_dir, "metrics.csv")
    
    if os.path.exists(metrics_csv_path):
        existing_metrics = pd.read_csv(metrics_csv_path)
        # Filter out existing records for this ticker + classical models to prevent duplication
        classical_model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'SVR']
        existing_metrics = existing_metrics[
            ~((existing_metrics['Ticker'] == ticker) & (existing_metrics['Model'].isin(classical_model_names)))
        ]
        combined_metrics = pd.concat([existing_metrics, metrics_df], ignore_index=True)
        combined_metrics.to_csv(metrics_csv_path, index=False)
    else:
        metrics_df.to_csv(metrics_csv_path, index=False)
        
    print(f"Finished classical training & validation for {ticker}. Saved metrics.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classical ML models.")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help="Tickers to process.")
    parser.add_argument('--processed_dir', default='data/processed', help="Directory containing processed data")
    parser.add_argument('--models_dir', default='models', help="Directory to save models")
    parser.add_argument('--outputs_dir', default='outputs', help="Directory to save metrics")
    
    args = parser.parse_args()
    for ticker in args.tickers:
        train_and_evaluate(ticker, args.processed_dir, args.models_dir, args.outputs_dir)
