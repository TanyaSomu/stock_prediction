# Project Log — ML Stock Market Forecasting System

## 1. Feature Registry

| Feature | What it is | Purpose / How it's used in the project | File(s) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Historical Data Collection** | Script downloading stock price histories (Open, High, Low, Close, Volume) using the yfinance API | Downloads raw data for training and testing ML models; saves them as raw CSV files | `1_data_collection.py` | Done |
| **Data Preprocessing & Scaling** | Cleans raw data, computes technical indicators, divides data chronologically (80/20), and fits StandardScaler objects | Prepares features and targets for training and evaluation; saves standard scalers for runtime inference | `2_preprocessing.py` | Done |
| **SMA (10, 50)** | Simple Moving Average over 10-day and 50-day windows | Used as trend features to help ML models identify short-term and medium-term price trends | `2_preprocessing.py` | Done |
| **EMA (10, 50)** | Exponential Moving Average placing higher weight on recent prices | Fed as input features to all 5 ML models to detect fast-moving price trends | `2_preprocessing.py` | Done |
| **RSI (14)** | Relative Strength Index, momentum oscillator measuring speed/change of price movements | Fed as input momentum feature to detect overbought (>70) or oversold (<30) market conditions | `2_preprocessing.py` | Done |
| **MACD & Signal** | Moving Average Convergence Divergence trend-following momentum indicator | Fed as input momentum features to identify changes in trend strength and direction | `2_preprocessing.py` | Done |
| **Bollinger Bands** | Volatility envelope defining upper, middle, and lower bands | Input features indicating price range bounds and volatility expansion/contraction | `2_preprocessing.py` | Done |
| **Asset Volatility** | 20-day rolling standard deviation of daily price returns | Input feature to detect stock risk; also used by recommendation engine to bucket asset risk levels | `2_preprocessing.py`, `recommendation.py` | Done |
| **Autoregressive Lags** | Past Close prices shifted by 1, 2, and 3 days (Lag_1, Lag_2, Lag_3) | Features feeding historical price context directly to models to capture short-term memory | `2_preprocessing.py` | Done |
| **Forecast Target** | Close price shifted by -1 day (representing t+1 Close) | Defines the labels for model training and represents the actual price to compare predictions against | `2_preprocessing.py` | Done |
| **VADER Sentiment Scores** | Lexicon-based sentiment scoring producing compound, positive, negative, and neutral scores | Evaluates market tone as an extra feature dimension to assess news correlation on forecasting | `sentiment.py` | Done |
| **Linear Regression** | Classical regression algorithm modeling linear relationships | Baseline model for comparing non-linear forecasting performance | `3_train_models.py` | Done |
| **Random Forest Regressor** | Ensemble tree-based regressor using bootstrap aggregation | Predicts t+1 price using non-linear combinations of price technicals and sentiment | `3_train_models.py` | Done |
| **XGBoost Regressor** | Gradient-boosted decision tree algorithm | Predicts t+1 price; high-performance model evaluated in SHAP explainability analyses | `3_train_models.py` | Done |
| **Support Vector Regression** | Support Vector Machine adapted for regression using RBF kernel | Predicts t+1 price; handles non-linear boundaries in higher-dimensional space | `3_train_models.py` | Done |
| **LSTM Model** | Deep learning recurrent neural network utilizing 60-day windows | Captures long-term sequential dependencies in price series to output t+1 predictions | `4_train_lstm.py` | Done |
| **Recommendation Engine** | Logic converting predicted returns and model confidences into Buy/Hold/Sell signals | Generates trading guidance logs based on whether predictions exceed thresholds and past errors | `recommendation.py` | Done |
| **Backtesting Simulator** | Compounding trading simulator that evaluates cumulative strategy returns against Buy & Hold | Tests the trading profitability of the models' recommendations over the test set | `backtest.py` | Done |
| **SHAP Explainability** | SHAP feature attribution library mapping model inputs to prediction impacts | Outputs global summary plots and local waterfall charts to explain tree model decisions | `explainability.py` | Done |
| **Model Evaluation & Plotting** | Script generating comparison charts (RMSE, R2, sentiment ablation, and strategy returns) | Consolidates metrics into publication-ready graphs saved in outputs/figures/ | `5_evaluate.py` | Done |
| **Unit Testing Suite** | Comprehensive unit test suite covering data preprocessing, LSTM sequence target alignment, and metric calculations | Verifies core algorithms and guards against regression and data alignment errors | `test_suite.py` | Done |
| **Test Cases Documentation** | Markdown file documenting the automated verification suite, expected behaviors, and run commands | Serves as a reference guide for project verification and review panels | `TEST_CASES.md` | Done |
| **Interactive Dashboard** | Premium Streamlit UI compiling all charts, models, sentiments, and evaluations | Serves as the user interface to pick assets, visualize actual vs. predicted curves, and view SHAP | `6_app.py` | Done |

---

## 2. Changelog

### 2026-08-17 — Implementation of Unit Testing Suite
- **What was changed and why:**
  - Designed and created a comprehensive testing suite `test_suite.py` to cover key pipeline functions.
  - Implemented unit test cases for indicators columns verification, value ranges, RSI division-by-zero robustness, LSTM feature-target alignment correctness, LSTM sequence dimensions, and MAPE metrics calculations.
  - Created a separate markdown documentation log `TEST_CASES.md` to map out each test case, its purpose, expected asserts, and execution instructions.
- **Files affected:**
  - `test_suite.py`
  - `TEST_CASES.md`
- **Results/Metrics changes:**
  - All 6 unit tests successfully ran and returned OK in 0.038 seconds, providing automated validation of the codebase.
- **Decisions made & reasoning:**
  - Chose the standard built-in `unittest` framework to avoid introducing external dependencies, and resolved number-prefixed module imports dynamically using Python's `importlib`. Documented the tests in a clean, user-friendly Markdown schema to support paper submissions and review panels.

### 2026-08-17 — Temporal Alignment, Look-Ahead Bias, and Stability Fixes
- **What was changed and why:**
  - **LSTM Alignment Bug:** Fixed `create_sequences` in `4_train_lstm.py` to map features up to index `i + seq_length - 1` to target `targets[i + seq_length - 1]` instead of `targets[i + seq_length]`. Adjusted the validation date slicing in `test_dates` to `test_df['Date'].values[seq_length - 1 : -1]`. This corrects an off-by-one alignment bug that was causing a 2-day-ahead forecast instead of a 1-day-ahead forecast and skipping the current day's features.
  - **Look-Ahead Bias in Recommendations:** Replaced the global median confidence threshold (`merged_df['model_confidence'].median()`) with a rolling 30-day historical median of past model confidences in `recommendation.py` to prevent future test set leakage into early trading decisions.
  - **RSI Division-by-Zero:** Added a small epsilon (`1e-9`) to the division in the RSI calculation inside `2_preprocessing.py` to prevent potential runtime NaN crashes.
  - **Evaluation Plot Crashes:** Changed `df.pivot()` to `df.pivot_table(..., aggfunc='mean')` in `5_evaluate.py` to make the script robust when duplicate model entries are present in `metrics.csv`.
  - **Naming Inconsistencies:** Updated filename parsing in `backtest.py` to keep the abbreviations `LSTM` and `SVR` in uppercase, resolving model naming mismatches in backtesting logs and metrics.
- **Files affected:** 
  - `2_preprocessing.py`
  - `4_train_lstm.py`
  - `recommendation.py`
  - `backtest.py`
  - `5_evaluate.py`
- **Results/Metrics changes:**
  - **LSTM performance:** The LSTM metrics changed because the model is now correctly trained to predict 1 day ahead (using the most recent day's data) rather than 2 days ahead.
  - **Trading signals:** The trading signal distributions for LSTM and other models shifted significantly since they are now guided by a realistic rolling historical confidence threshold rather than a global future median. For example, AAPL LSTM (With Sentiment) recommendation distribution changed from its prior state to Buy: 0, Hold: 360, Sell: 155, showing more realistic and risk-averse trading signals.
- **Decisions made & reasoning:**
  - Implemented a 30-day historical rolling window for confidence thresholding. The 30-day period was selected because it represents approximately 1.5 calendar months of trading, providing a sample of recent model prediction errors without carrying stale errors from months ago.

### 2026-08-16 — Initial System Implementation (Retroactive Entry)
- **What was changed and why:**
  - Set up the entire stock forecasting pipeline, including data downloading, preprocessing, training classical ML models (Linear Regression, Random Forest, XGBoost, SVR), training a sequential deep learning model (LSTM), creating a signal recommendation logic, executing backtests, running SHAP explainability analyses, and constructing the interactive Streamlit dashboard.
- **Files affected:**
  - `1_data_collection.py`
  - `2_preprocessing.py`
  - `3_train_models.py`
  - `4_train_lstm.py`
  - `sentiment.py`
  - `recommendation.py`
  - `backtest.py`
  - `explainability.py`
  - `5_evaluate.py`
  - `6_app.py`
- **Results/Metrics changes:**
  - Baseline metrics established and stored in `outputs/metrics.csv` for AAPL, MSFT, and GOOGL.
- **Decisions made & reasoning:**
  - Selected VADER over advanced Transformers (e.g. FinBERT) for the initial sentiment pipeline setup because VADER relies on rules and lexicons, allowing offline execution without needing GPU resources or incurring high latency.

---

## 3. Known Issues / Open Questions

1. **Simulated Sentiment Training Discrepancy:**
   - Sentiment scores in the training set are simulated directly from the daily returns to establish text correlation during backtests. However, if `NEWS_API_KEY` is active, the test set uses real news headlines. Because the free tier of NewsAPI limits fetches to the last 30 days, 90%+ of the test set dates fall back to simulated headlines. This creates a data inconsistency where models trained on return-correlated sentiment are exposed to noisier real news sentiment.
2. **SHAP Attributions Only for Tree-based Models:**
   - SHAP feature analysis is only implemented for Random Forest and XGBoost using `TreeExplainer`. SHAP values for the LSTM model are not yet integrated, as `KernelExplainer` or `DeepExplainer` can be computationally expensive on CPU-only machines.
3. **Data Collection yfinance Column Variations:**
   - Newer versions of the `yfinance` package sometimes structure multi-index headers differently depending on the number of tickers queried. The downloader script handles this with index splitting, but further structure modifications in `yfinance` could break the columns.

---

## 4. Decisions Log

### Chronological 80/20 Train/Test Split
* **Decision:** Used a chronological index split (first 80% for training, last 20% for testing) instead of random k-fold cross-validation.
* **Reasoning:** Standard random splits/k-fold cross-validations are inappropriate for time-series forecasting. They leak future price information into the past, causing look-ahead bias and giving a false impression of model accuracy.
* **Methodology Impact:** Guarantees that evaluation results closely mimic live out-of-sample deployment.

### Scaling fitted strictly on Training Data
* **Decision:** Fitted the `StandardScaler` objects strictly on the training partition and only applied `.transform()` to the test partition.
* **Reasoning:** Fitting the scaler on the entire dataset would leak the mean and variance of the test set into the training phase, violating out-of-sample testing principles.

### Rolling confidence threshold vs. Global median
* **Decision:** Changed trading recommendations to use a 30-day rolling historical median of model confidence instead of a static test set median.
* **Reasoning:** The global test set median relies on knowledge of all future errors (spanning years), introducing a severe look-ahead data leakage. A rolling historical median ensures the model makes trading decisions on day $t$ using only information available up to day $t$.

### Rule-based Lexicon Sentiment (VADER)
* **Decision:** Selected VADER Sentiment Intensity Analyzer as the sentiment scoring module.
* **Reasoning:** VADER is lightweight, offline-capable, and optimized for social/financial text snippets. It can be easily integrated without requiring high-performance deep learning servers, making the data pipeline lightweight and portable.
