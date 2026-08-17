import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_model(ticker, rec_file, models_dir, figures_dir):
    """
    Simulates trading based on recommendations and evaluates performance.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    rec_path = os.path.join(models_dir, rec_file)
    if not os.path.exists(rec_path):
        print(f"Error: Recommendation file {rec_path} not found.")
        return None
        
    df = pd.read_csv(rec_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Map recommendation to signal
    # 1 = Buy, 0 = Hold, -1 = Sell (Short)
    signal_map = {'Buy': 1, 'Hold': 0, 'Sell': -1}
    df['Signal'] = df['Recommendation'].map(signal_map)
    
    # Calculate daily strategy return: signal on day t * return on day t+1
    # Note: actual_return_pct is the return from day t to day t+1
    df['Strategy_Return'] = df['Signal'] * df['actual_return_pct']
    
    # Cumulative returns
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1
    df['Cum_BuyHold_Return'] = (1 + df['actual_return_pct']).cumprod() - 1
    
    # Performance Metrics
    strategy_returns = df['Strategy_Return'].values
    buyhold_returns = df['actual_return_pct'].values
    
    # Sharpe Ratio (annualized, 252 trading days)
    def calculate_sharpe(returns):
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        return (mean_ret / std_ret) * np.sqrt(252)
        
    strat_sharpe = calculate_sharpe(strategy_returns)
    bh_sharpe = calculate_sharpe(buyhold_returns)
    
    strat_cum = df['Cum_Strategy_Return'].iloc[-1]
    bh_cum = df['Cum_BuyHold_Return'].iloc[-1]
    
    # Parse model details from filename
    parts = rec_file.replace('_recommendations.csv', '').split('_')
    if 'sentiment' in parts:
        suffix = '_sentiment'
        model_name_parts = parts[1:-1]
        features_label = "With Sentiment"
    else:
        suffix = ''
        model_name_parts = parts[1:]
        features_label = "Without Sentiment"
        
    model_name = ' '.join([p.upper() if p.lower() in ['lstm', 'svr'] else p.capitalize() for p in model_name_parts])
    
    # Save backtest data
    backtest_data_path = os.path.join(models_dir, rec_file.replace('_recommendations.csv', '_backtest.csv'))
    df.to_csv(backtest_data_path, index=False)
    
    # Plot equity curve comparison
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Cum_Strategy_Return'] * 100, label=f"Strategy ({model_name} {features_label})", color='#2ca02c', linewidth=2)
    plt.plot(df['Date'], df['Cum_BuyHold_Return'] * 100, label="Buy & Hold (Benchmark)", color='#1f77b4', linestyle='--', linewidth=1.5)
    
    plt.title(f"Backtesting Equity Curve: {ticker} ({model_name} - {features_label})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    
    plot_file = f"{ticker}_{model_name.lower().replace(' ', '_')}{suffix}_backtest.png"
    plot_path = os.path.join(figures_dir, plot_file)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Backtest Completed for {ticker} - {model_name} ({features_label}):")
    print(f"  Strategy Cumulative Return: {strat_cum * 100:.2f}% | Sharpe: {strat_sharpe:.2f}")
    print(f"  Buy & Hold Cumulative Return: {bh_cum * 100:.2f}% | Sharpe: {bh_sharpe:.2f}")
    print(f"  Equity curve plot saved to {plot_path}")
    
    return {
        'Ticker': ticker,
        'Model': model_name,
        'Features': features_label,
        'Strategy_Return_Cum': strat_cum,
        'Strategy_Sharpe': strat_sharpe,
        'BuyHold_Return_Cum': bh_cum,
        'BuyHold_Sharpe': bh_sharpe
    }

def run_all_backtests(tickers, processed_dir, models_dir, figures_dir):
    """
    Runs backtesting for all tickers and recommendation files found.
    """
    results = []
    for ticker in tickers:
        rec_files = [f for f in os.listdir(models_dir) if f.startswith(ticker) and f.endswith('_recommendations.csv')]
        for rec_file in rec_files:
            res = backtest_model(ticker, rec_file, models_dir, figures_dir)
            if res:
                results.append(res)
                
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(models_dir, "backtest_summary.csv"), index=False)
        print("\nConsolidated backtest summary saved to models/backtest_summary.csv.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backtest trading recommendations.")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help="Tickers to process.")
    parser.add_argument('--processed_dir', default='data/processed', help="Directory containing processed data")
    parser.add_argument('--models_dir', default='models', help="Directory containing recommendations")
    parser.add_argument('--figures_dir', default='outputs/figures', help="Directory to save backtest charts")
    
    args = parser.parse_args()
    run_all_backtests(args.tickers, args.processed_dir, args.models_dir, args.figures_dir)
