import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """
    Sets a publication-quality style for Matplotlib charts.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'font.family': 'sans-serif'
    })

def plot_model_comparison(metrics_path, figures_dir):
    """
    Plots a bar chart comparing RMSE and R2 of all models (With vs. Without Sentiment).
    """
    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} not found. Train models first.")
        return
        
    df = pd.read_csv(metrics_path)
    
    # 1. RMSE comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y='RMSE', hue='Features', palette='viridis')
    plt.title("Model Comparison: Root Mean Squared Error (RMSE) in USD", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Model Type", fontsize=12)
    plt.ylabel("RMSE ($)", fontsize=12)
    plt.legend(title="Features", loc="upper right")
    plt.tight_layout()
    
    rmse_plot_path = os.path.join(figures_dir, "model_comparison_rmse.png")
    plt.savefig(rmse_plot_path, dpi=300)
    plt.close()
    print(f"Saved model comparison RMSE plot to {rmse_plot_path}")
    
    # 2. R2 comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y='R2', hue='Features', palette='mako')
    plt.title("Model Comparison: R-squared (R2) Score", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Model Type", fontsize=12)
    plt.ylabel("R-squared (Higher is Better)", fontsize=12)
    plt.legend(title="Features", loc="lower right")
    plt.tight_layout()
    
    r2_plot_path = os.path.join(figures_dir, "model_comparison_r2.png")
    plt.savefig(r2_plot_path, dpi=300)
    plt.close()
    print(f"Saved model comparison R2 plot to {r2_plot_path}")

def plot_sentiment_ablation(metrics_path, figures_dir):
    """
    Plots an ablation study showing percentage reduction in RMSE when sentiment is added.
    """
    if not os.path.exists(metrics_path):
        return
        
    df = pd.read_csv(metrics_path)
    
    # Group by ticker, model and compute the difference (handling duplicates robustly)
    pivot_df = df.pivot_table(index=['Ticker', 'Model'], columns='Features', values='RMSE', aggfunc='mean').reset_index()
    if 'With Sentiment' not in pivot_df.columns or 'Without Sentiment' not in pivot_df.columns:
        print("Ablation study requires both 'With Sentiment' and 'Without Sentiment' metrics.")
        return
        
    # Calculate % improvement (error reduction)
    pivot_df['RMSE_Reduction_Pct'] = ((pivot_df['Without Sentiment'] - pivot_df['With Sentiment']) / pivot_df['Without Sentiment']) * 100
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=pivot_df, x='Model', y='RMSE_Reduction_Pct', hue='Ticker', palette='Set2')
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    
    plt.title("Ablation Study: Error Reduction (%) with Sentiment Features", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Model Type", fontsize=12)
    plt.ylabel("RMSE Decrease (%)", fontsize=12)
    plt.legend(title="Stock Ticker")
    plt.tight_layout()
    
    ablation_plot_path = os.path.join(figures_dir, "sentiment_ablation_study.png")
    plt.savefig(ablation_plot_path, dpi=300)
    plt.close()
    print(f"Saved sentiment ablation study plot to {ablation_plot_path}")

def plot_actual_vs_predicted(tickers, models_dir, figures_dir, target_model="XGBoost", with_sentiment=True):
    """
    Plots actual vs predicted line plots for the test set of each ticker.
    """
    suffix = "_sentiment" if with_sentiment else ""
    feat_lbl = "With Sentiment" if with_sentiment else "Without Sentiment"
    
    for ticker in tickers:
        pred_file = f"{ticker}_{target_model.lower().replace(' ', '_')}{suffix}_predictions.csv"
        pred_path = os.path.join(models_dir, pred_file)
        
        if not os.path.exists(pred_path):
            print(f"Predictions for {ticker} using {target_model} ({feat_lbl}) not found. Skipping plot.")
            continue
            
        df = pd.read_csv(pred_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Plot last 100 days for visual clarity
        plot_df = df.tail(100)
        
        plt.figure(figsize=(12, 5))
        plt.plot(plot_df['Date'], plot_df['Actual'], label="Actual Price", color='#1f77b4', linewidth=2)
        plt.plot(plot_df['Date'], plot_df['Predicted'], label="Predicted Price", color='#ff7f0e', linestyle='--', linewidth=2)
        
        plt.title(f"Actual vs. Predicted Price Over Test Set: {ticker} ({target_model} {feat_lbl})", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Date", fontsize=11)
        plt.ylabel("Stock Close Price ($)", fontsize=11)
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        
        output_file = f"{ticker}_{target_model.lower().replace(' ', '_')}{suffix}_actual_vs_pred.png"
        output_path = os.path.join(figures_dir, output_file)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved actual vs. predicted plot to {output_path}")

def plot_backtest_performance(models_dir, figures_dir):
    """
    Plots cumulative return comparison of all backtested strategies.
    """
    summary_path = os.path.join(models_dir, "backtest_summary.csv")
    if not os.path.exists(summary_path):
        print("Backtest summary not found. Run backtest.py first.")
        return
        
    df = pd.read_csv(summary_path)
    
    # Plot strategy cumulative returns vs buy-and-hold returns
    plt.figure(figsize=(12, 6))
    
    # Shorten feature text for visualization
    df['Label'] = df['Model'] + " (" + df['Features'].replace({'With Sentiment': 'Sent', 'Without Sentiment': 'NoSent'}) + ")"
    
    # Sort for neat plotting
    df_sorted = df.sort_values('Strategy_Return_Cum', ascending=False)
    
    sns.barplot(data=df_sorted, x='Label', y='Strategy_Return_Cum', hue='Ticker', palette='crest')
    plt.axhline(0, color='black', linewidth=0.8)
    
    plt.title("Strategy Cumulative Returns comparison Across Models", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Model Configuration", fontsize=12)
    plt.ylabel("Cumulative Return (Decimal)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    backtest_chart_path = os.path.join(figures_dir, "backtest_performance_comparison.png")
    plt.savefig(backtest_chart_path, dpi=300)
    plt.close()
    print(f"Saved backtest performance comparison to {backtest_chart_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate publication-ready figures.")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help="Tickers to process.")
    parser.add_argument('--models_dir', default='models', help="Directory containing prediction files")
    parser.add_argument('--outputs_dir', default='outputs', help="Directory with metrics.csv")
    parser.add_argument('--figures_dir', default='outputs/figures', help="Directory to save figures")
    
    args = parser.parse_args()
    
    set_style()
    metrics_csv = os.path.join(args.outputs_dir, "metrics.csv")
    
    plot_model_comparison(metrics_csv, args.figures_dir)
    plot_sentiment_ablation(metrics_csv, args.figures_dir)
    plot_actual_vs_predicted(args.tickers, args.models_dir, args.figures_dir, target_model="XGBoost", with_sentiment=True)
    plot_actual_vs_predicted(args.tickers, args.models_dir, args.figures_dir, target_model="LSTM", with_sentiment=True)
    plot_backtest_performance(args.models_dir, args.figures_dir)
    
    print("\nAll evaluation figures generated successfully in outputs/figures/.")
