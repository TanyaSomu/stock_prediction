import os
import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime

def collect_data(tickers, start_date, end_date, output_dir):
    """
    Downloads historical stock data using yfinance and saves to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting data collection from {start_date} to {end_date}...")
    
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            # Download data from yfinance
            df = yf.download(ticker, start=start_date, end=end_date)
            
            if df.empty:
                print(f"Warning: No data returned for {ticker}.")
                continue
                
            # Flatten multi-level columns if present (can happen in some yfinance outputs)
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten the MultiIndex columns
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
                # If yfinance returned columns like "Close_AAPL", rename them to "Close"
                df.columns = [col.split('_')[0] for col in df.columns]
            
            # Reset index so 'Date' is a column
            df = df.reset_index()
            
            # Ensure column names are standard: Date, Open, High, Low, Close, Adj Close, Volume
            # Some versions of yfinance might name Adj Close differently
            rename_dict = {}
            for col in df.columns:
                if 'adj' in col.lower() and 'close' in col.lower():
                    rename_dict[col] = 'Adj Close'
                elif 'open' in col.lower():
                    rename_dict[col] = 'Open'
                elif 'high' in col.lower():
                    rename_dict[col] = 'High'
                elif 'low' in col.lower():
                    rename_dict[col] = 'Low'
                elif 'close' in col.lower():
                    rename_dict[col] = 'Close'
                elif 'volume' in col.lower():
                    rename_dict[col] = 'Volume'
                elif 'date' in col.lower():
                    rename_dict[col] = 'Date'
            
            df = df.rename(columns=rename_dict)
            
            # Keep only standard columns if they exist
            cols_to_keep = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
            df = df[cols_to_keep]
            
            # Save raw data
            output_path = os.path.join(output_dir, f"{ticker}.csv")
            df.to_csv(output_path, index=False)
            print(f"Successfully saved raw data for {ticker} to {output_path} ({len(df)} rows).")
            
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download stock data from Yahoo Finance.")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help="List of stock tickers to download.")
    parser.add_argument('--start', default='2015-01-01', help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help="End date (YYYY-MM-DD)")
    parser.add_argument('--output_dir', default='data/raw', help="Directory to save raw data")
    
    args = parser.parse_args()
    collect_data(args.tickers, args.start, args.end, args.output_dir)
