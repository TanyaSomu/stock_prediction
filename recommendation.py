import os
import argparse
import pandas as pd
import numpy as np
import joblib

def generate_recommendations(ticker, processed_dir, models_dir):
    """
    Computes predicted return, model confidence, and risk level.
    Generates trading recommendations (Buy/Hold/Sell) for all models.
    """
    # Load training unscaled data to compute risk (volatility) thresholds without data leakage
    train_path = os.path.join(processed_dir, f"{ticker}_train_unscaled_sentiment.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join(processed_dir, f"{ticker}_train_unscaled.csv")
        
    if not os.path.exists(train_path):
        print(f"Error: Training file not found for {ticker}")
        return
        
    train_df = pd.read_csv(train_path)
    train_vols = train_df['Volatility'].dropna().values
    
    # Define volatility thresholds based on training data quantiles
    vol_q33 = np.percentile(train_vols, 33)
    vol_q66 = np.percentile(train_vols, 66)
    
    # Load unscaled test data for current price (Close) and Volatility alignment
    test_unscaled_path = os.path.join(processed_dir, f"{ticker}_test_unscaled_sentiment.csv")
    if not os.path.exists(test_unscaled_path):
        test_unscaled_path = os.path.join(processed_dir, f"{ticker}_test_unscaled.csv")
        
    test_unscaled = pd.read_csv(test_unscaled_path)
    
    # Find all prediction files in models/ directory for this ticker
    prediction_files = [f for f in os.listdir(models_dir) if f.startswith(ticker) and f.endswith('_predictions.csv')]
    
    for pred_file in prediction_files:
        # Extract model details from filename
        # Format is typically: {ticker}_{model_name}_{sentiment}_predictions.csv
        parts = pred_file.replace('_predictions.csv', '').split('_')
        ticker_part = parts[0]
        
        # Reconstruct model name and suffix (sentiment or not)
        # e.g., ['AAPL', 'linear', 'regression', 'sentiment']
        if 'sentiment' in parts:
            suffix = '_sentiment'
            model_name_parts = parts[1:-1]
        else:
            suffix = ''
            model_name_parts = parts[1:]
            
        model_name = ' '.join([p.capitalize() for p in model_name_parts])
        
        pred_path = os.path.join(models_dir, pred_file)
        preds_df = pd.read_csv(pred_path)
        
        # Merge with unscaled test data to get Close and Volatility
        merged_df = pd.merge(preds_df, test_unscaled[['Date', 'Close', 'Volatility']], on='Date', how='inner')
        merged_df = merged_df.sort_values('Date').reset_index(drop=True)
        
        # Compute predicted return pct: (Predicted_Close_t+1 - Close_t) / Close_t
        merged_df['predicted_return_pct'] = (merged_df['Predicted'] - merged_df['Close']) / merged_df['Close']
        merged_df['actual_return_pct'] = (merged_df['Actual'] - merged_df['Close']) / merged_df['Close']
        
        # Compute model confidence: inverse of 20-day rolling RMSE
        merged_df['error'] = merged_df['Predicted'] - merged_df['Actual']
        merged_df['rolling_rmse'] = np.sqrt((merged_df['error']**2).rolling(window=20, min_periods=1).mean())
        merged_df['model_confidence'] = 1.0 / np.where(merged_df['rolling_rmse'] == 0, 1e-5, merged_df['rolling_rmse'])
        
        # Volatility risk bucketing
        def bucket_volatility(vol):
            if vol < vol_q33:
                return "Low"
            elif vol < vol_q66:
                return "Medium"
            else:
                return "High"
                
        merged_df['risk_level'] = merged_df['Volatility'].apply(bucket_volatility)
        
        # Recommendation logic
        # Set return thresholds: buy if predicted return > +0.1%, sell if < -0.1%
        buy_threshold = 0.001
        sell_threshold = -0.001
        
        # Compute confidence threshold dynamically using 30-day rolling median of past confidence to prevent look-ahead bias
        merged_df['conf_threshold'] = merged_df['model_confidence'].shift(1).rolling(window=30, min_periods=1).median()
        merged_df['conf_threshold'] = merged_df['conf_threshold'].bfill().fillna(0)
        
        def generate_label(row):
            ret = row['predicted_return_pct']
            conf = row['model_confidence']
            thresh = row['conf_threshold']
            
            if ret > buy_threshold and conf >= thresh:
                return "Buy"
            elif ret < sell_threshold and conf >= thresh:
                return "Sell"
            else:
                return "Hold"
                
        merged_df['Recommendation'] = merged_df.apply(generate_label, axis=1)
        
        # Save recommendations file
        rec_file_name = pred_file.replace('_predictions.csv', '_recommendations.csv')
        rec_save_path = os.path.join(models_dir, rec_file_name)
        
        # Keep relevant columns
        output_cols = [
            'Date', 'Close', 'Actual', 'Predicted', 'predicted_return_pct', 
            'actual_return_pct', 'model_confidence', 'rolling_rmse', 'Volatility', 'risk_level', 'Recommendation'
        ]
        merged_df[output_cols].to_csv(rec_save_path, index=False)
        
        # Count signals
        counts = merged_df['Recommendation'].value_counts()
        print(f"Generated recommendations for {ticker} - {model_name} ({'With' if suffix else 'Without'} Sentiment):")
        print(f"  Saved to {rec_save_path}")
        print(f"  Signal distribution -> Buy: {counts.get('Buy', 0)}, Hold: {counts.get('Hold', 0)}, Sell: {counts.get('Sell', 0)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate trading recommendations.")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help="Tickers to process.")
    parser.add_argument('--processed_dir', default='data/processed', help="Directory containing processed data")
    parser.add_argument('--models_dir', default='models', help="Directory with predictions and models")
    
    args = parser.parse_args()
    for ticker in args.tickers:
        generate_recommendations(ticker, args.processed_dir, args.models_dir)
