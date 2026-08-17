import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

def generate_explanations(ticker, processed_dir, models_dir, figures_dir):
    """
    Applies SHAP to RF and XGBoost models (with sentiment) to explain predictions.
    Outputs global feature importance plots and a local waterfall explanation plot.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # We explain the "With Sentiment" models to demonstrate feature impact of text analysis
    suffix = "_sentiment"
    test_path = os.path.join(processed_dir, f"{ticker}_test_scaled{suffix}.csv")
    
    if not os.path.exists(test_path):
        print(f"Preprocessed test data not found at {test_path}. Run sentiment analysis first.")
        return
        
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=['Date', 'Target'])
    
    # Models to analyze
    model_names = ['Random Forest', 'XGBoost']
    
    for name in model_names:
        model_file = f"{ticker}_{name.lower().replace(' ', '_')}{suffix}.joblib"
        model_path = os.path.join(models_dir, model_file)
        
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Skipping explainability for {name}.")
            continue
            
        print(f"Computing SHAP values for {ticker} - {name}...")
        model = joblib.load(model_path)
        
        # Initialize TreeExplainer (optimized for tree-based models like RF and XGBoost)
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values on test set
        shap_values = explainer.shap_values(X_test)
        
        # Handle cases where shap_values might have an extra dimension or be a list (e.g. multi-class, but this is regression)
        if isinstance(shap_values, list):
            shap_val_array = shap_values[0]
        else:
            shap_val_array = shap_values
            
        # 1. Global Summary Plot
        plt.figure(figsize=(10, 6))
        # Clear current figure to avoid overlapping plots
        plt.clf()
        shap.summary_plot(shap_val_array, X_test, show=False)
        plt.title(f"SHAP Global Feature Importance: {name} ({ticker})", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        
        global_plot_path = os.path.join(figures_dir, f"{ticker}_{name.lower().replace(' ', '_')}_shap_global.png")
        plt.savefig(global_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved global SHAP summary plot to {global_plot_path}")
        
        # 2. Local Explanation Plot (for the first instance in test set)
        plt.figure(figsize=(10, 5))
        plt.clf()
        
        # Create an Explanation object which waterfall plot expects in modern SHAP versions
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 0:
            base_val = base_val[0]
            
        single_shap = shap_val_array[0]
        
        explanation = shap.Explanation(
            values=single_shap,
            base_values=base_val,
            data=X_test.iloc[0].values,
            feature_names=X_test.columns.tolist()
        )
        
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.title(f"SHAP Local Explanation (First Test Sample): {name} ({ticker})", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        
        local_plot_path = os.path.join(figures_dir, f"{ticker}_{name.lower().replace(' ', '_')}_shap_local.png")
        plt.savefig(local_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved local SHAP explanation plot to {local_plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate SHAP model explanations.")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help="Tickers to process.")
    parser.add_argument('--processed_dir', default='data/processed', help="Directory containing processed data")
    parser.add_argument('--models_dir', default='models', help="Directory with trained models")
    parser.add_argument('--figures_dir', default='outputs/figures', help="Directory to save SHAP figures")
    
    args = parser.parse_args()
    for ticker in args.tickers:
        generate_explanations(ticker, args.processed_dir, args.models_dir, args.figures_dir)
