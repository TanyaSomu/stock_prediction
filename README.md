# ML-Based Stock Market Forecasting System

This project is a complete, end-to-end Machine Learning pipeline for Stock Market Forecasting, designed for a final-year engineering project. It downloads daily stock price data, engineers technical indicators, incorporates textual news sentiment via VADER NLP, trains classical and deep learning models (Linear Regression, Random Forest, XGBoost, Support Vector Regression, and LSTM), applies SHAP for explainability, generates trading recommendations, and performs backtesting. Finally, it visualizes all outputs on an interactive dark-themed Streamlit dashboard.

---

## Technical Features

1.  **Chronological Split**: Ensures time-series validity by splitting train/test sets chronologically (80/20) instead of randomly, preventing look-ahead bias.
2.  **Data Leakage Prevention**: Fits feature and target scalers strictly on the training set, then transforms both the training and test sets.
3.  **NLP News Sentiment Integration**: Computes daily VADER compound sentiment indices from news headlines (integrating with NewsAPI for recent news and generating correlated headlines for historical backtesting).
4.  **SHAP Explainability**: Explains the tree-based models (Random Forest and XGBoost) globally (summary plots) and locally (waterfall charts for individual predictions).
5.  **Backtesting Engine**: Simulates long-short trading based on model confidence and predicted return pct, generating cumulative strategy return curves vs. Buy & Hold benchmark.

---

## Directory Structure

```
d:/final-year project/ml-files/
├── data/
│   ├── raw/                 # Downloaded raw stock data (CSV)
│   └── processed/           # Engineered features and train/test splits (CSV)
├── models/                  # Saved models (joblib/h5), scalers, and predictions
├── outputs/
│   ├── figures/             # High-resolution publication-quality charts (PNG)
│   └── metrics.csv          # Single consolidated model performance comparison
├── 1_data_collection.py     # Downloads yfinance daily OHLCV data
├── 2_preprocessing.py       # Cleans data, builds indicators, splits, and scales
├── sentiment.py             # Pulls news/simulates sentiment, runs VADER, scales sentiment
├── 3_train_models.py        # Trains classical models (LR, RF, XGB, SVR)
├── 4_train_lstm.py          # Trains sequence-based LSTM
├── explainability.py        # Computes SHAP values and plots
├── recommendation.py        # Combines predictions and risk to generate Buy/Hold/Sell signals
├── backtest.py              # Runs backtests and calculates Sharpe ratios
├── 5_evaluate.py            # Generates consolidated publication charts
├── 6_app.py                 # Streamlit web app
└── requirements.txt         # Pinned python packages
```

---

## Run Order Sequence

To run the complete pipeline and generate all assets, execute the scripts in the following order:

### 1. Data Collection
Downloads stock data for configurable tickers (default: AAPL, MSFT, GOOGL) from 2015-01-01 to present.
```bash
python 1_data_collection.py
```
*   *Produces:* `data/raw/{ticker}.csv`

### 2. Preprocessing & Technical Indicators
Cleans data, creates chronological train/test splits, computes technical features (SMA, EMA, RSI, MACD, Bollinger Bands, lag variables, and Volatility), and scales features/targets.
```bash
python 2_preprocessing.py
```
*   *Produces:* `data/processed/{ticker}_train_scaled.csv`, `data/processed/{ticker}_test_scaled.csv`, and saved scalers in `models/`.

### 3. Sentiment Integration
Scores financial news headlines using NLTK VADER and merges them. Uses an optional environment variable `NEWS_API_KEY` for recent news, falling back to a correlated news simulator for historical ranges.
```bash
python sentiment.py
```
*   *Produces:* `data/processed/{ticker}_train_scaled_sentiment.csv`, `data/processed/{ticker}_test_scaled_sentiment.csv`

### 4. Train Classical Models
Trains Linear Regression, Random Forest, XGBoost, and SVR.
```bash
python 3_train_models.py
```
*   *Produces:* Saved model weights, test prediction arrays (`models/{ticker}_{model}_predictions.csv`), and initial `outputs/metrics.csv`.

### 5. Train Deep Learning LSTM Model
Trains a stacked 60-day sequence LSTM model.
```bash
python 4_train_lstm.py
```
*   *Produces:* Saved LSTM model `.h5` files and appends LSTM results to `outputs/metrics.csv`.

### 6. Explainability (SHAP)
Applies SHAP to the Random Forest and XGBoost models.
```bash
python explainability.py
```
*   *Produces:* Global summary charts and local waterfall charts saved in `outputs/figures/`.

### 7. Generate Recommendations
Generates risk-adjusted Buy/Hold/Sell recommendation labels based on predicted return and confidence (inverse of rolling RMSE).
```bash
python recommendation.py
```
*   *Produces:* `models/{ticker}_{model}_recommendations.csv`

### 8. Backtesting
Converts recommendation signals into trading decisions and simulates strategy cumulative returns.
```bash
python backtest.py
```
*   *Produces:* Cumulative return logs (`models/{ticker}_{model}_backtest.csv`) and equity comparison charts in `outputs/figures/`.

### 9. Evaluate & Plot Unified Figures
Generates publication-quality figures for paper results sections (comparison bar charts, sentiment impact/ablation charts, backtesting metrics comparison).
```bash
python 5_evaluate.py
```
*   *Produces:* Consolidated charts in `outputs/figures/`.

### 10. Start Streamlit Dashboard
Launches the interactive dashboard.
```bash
streamlit run 6_app.py
```
