import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set Streamlit Page Config
st.set_page_config(
    page_title="FinAI: Stock Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark glassmorphism aesthetic)
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    /* Title Styling */
    .dashboard-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4d, #f68080, #3d84ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: #8a9ba8;
        margin-bottom: 2rem;
    }
    
    /* Card Container */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 75, 77, 0.4);
        box-shadow: 0 12px 40px 0 rgba(255, 75, 77, 0.15);
    }
    
    /* Recommendations Labels */
    .rec-buy {
        font-size: 2.5rem;
        font-weight: 800;
        color: #00e676;
        text-shadow: 0 0 15px rgba(0, 230, 118, 0.4);
        margin: 10px 0;
    }
    .rec-sell {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ff1744;
        text-shadow: 0 0 15px rgba(255, 23, 68, 0.4);
        margin: 10px 0;
    }
    .rec-hold {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffea00;
        text-shadow: 0 0 15px rgba(255, 234, 0, 0.4);
        margin: 10px 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        color: #8a9ba8;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f0f2f6;
        background-color: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 75, 77, 0.15) !important;
        color: #ff4b4d !important;
        border-color: rgba(255, 75, 77, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown('<div class="dashboard-title">FinAI: AI Stock Forecast & Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Machine Learning for Stock Market Forecasting: Trends and Techniques (Dept. of IT, AI Domain)</div>', unsafe_allow_html=True)

# Sidebar settings
st.sidebar.image("https://img.icons8.com/nolan/96/stock-share.png", width=80)
st.sidebar.markdown("### Control Panel")

# Stock Picker
tickers = ['AAPL', 'MSFT', 'GOOGL']
selected_ticker = st.sidebar.selectbox("Select Asset Ticker", tickers)

# Helper function to load datasets
@st.cache_data
def load_processed_data(ticker, dataset_type="test_unscaled_sentiment"):
    path = f"data/processed/{ticker}_{dataset_type}.csv"
    if not os.path.exists(path):
        # Fallback to no sentiment
        path = f"data/processed/{ticker}_{dataset_type.replace('_sentiment', '')}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Load unscaled test data for charts
test_df = load_processed_data(selected_ticker, "test_unscaled_sentiment")

# Sidebar - Display selected Ticker general info
if test_df is not None:
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    last_row = test_df.iloc[-1]
    prev_row = test_df.iloc[-2]
    curr_price = last_row['Close']
    price_change = curr_price - prev_row['Close']
    pct_change = (price_change / prev_row['Close']) * 100
    
    st.sidebar.markdown(f"## **${curr_price:.2f}**")
    st.sidebar.metric(
        label="Daily Price Action", 
        value=f"${curr_price:.2f}", 
        delta=f"{price_change:+.2f} ({pct_change:+.2f}%)"
    )
    st.sidebar.markdown(f"**Data Range:** {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")
else:
    st.sidebar.warning("Pipeline data not loaded. Run data collection and preprocessing scripts.")

# Tabs configuration
tabs = st.tabs([
    "📊 Historical Trends", 
    "📈 Predictions", 
    "🎯 Trading recommendations", 
    "⚔️ Backtest Results", 
    "🧠 Sentiment Overview", 
    "🏆 Model Comparison"
])

# -----------------
# TAB 1: HISTORICAL TRENDS
# -----------------
with tabs[0]:
    st.subheader("Historical Stock Price Trends & Indicators")
    if test_df is not None:
        # Date selection filter
        date_range = st.date_input(
            "Filter Date Range",
            value=(test_df['Date'].min().to_pydatetime(), test_df['Date'].max().to_pydatetime()),
            min_value=test_df['Date'].min().to_pydatetime(),
            max_value=test_df['Date'].max().to_pydatetime()
        )
        
        if len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered_df = test_df[(test_df['Date'] >= start_date) & (test_df['Date'] <= end_date)]
        else:
            filtered_df = test_df
            
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Interactive Candlestick Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=filtered_df['Date'],
                open=filtered_df['Open'],
                high=filtered_df['High'],
                low=filtered_df['Low'],
                close=filtered_df['Close'],
                name="OHLC Price"
            ))
            
            # Add Technical Indicators
            fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['SMA_10'], name="SMA 10", line=dict(color='#ff9800', width=1.5)))
            fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['SMA_50'], name="SMA 50", line=dict(color='#e91e63', width=1.5)))
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['BB_Upper'], name="BB Upper", line=dict(color='rgba(173, 216, 230, 0.4)', dash='dash')))
            fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['BB_Lower'], name="BB Lower", line=dict(color='rgba(173, 216, 230, 0.4)', dash='dash'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.05)'))
            
            fig.update_layout(
                title=f"{selected_ticker} Interactive Chart (OHLC, SMA, Bollinger Bands)",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=500,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI Chart below
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['RSI_14'], name="RSI (14)", line=dict(color='#ab47bc', width=2)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(
                title=f"{selected_ticker} Relative Strength Index (RSI 14)",
                template="plotly_dark",
                yaxis=dict(range=[0, 100]),
                height=200,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### Technical Summary")
            st.markdown(f"**Latest Close:** ${last_row['Close']:.2f}")
            st.markdown(f"**SMA(10):** ${last_row['SMA_10']:.2f}")
            st.markdown(f"**SMA(50):** ${last_row['SMA_50']:.2f}")
            
            rsi_val = last_row['RSI_14']
            rsi_status = "Neutral"
            if rsi_val > 70: rsi_status = "🔴 Overbought"
            elif rsi_val < 30: rsi_status = "🟢 Oversold"
            st.markdown(f"**RSI(14):** {rsi_val:.2f} ({rsi_status})")
            
            macd_val = last_row['MACD']
            macd_sig = last_row['MACD_Signal']
            macd_status = "Bullish Crossover" if macd_val > macd_sig else "Bearish Crossover"
            st.markdown(f"**MACD:** {macd_val:.3f} | Signal: {macd_sig:.3f} ({macd_status})")
            
            st.markdown(f"**Volatility (20D):** {last_row['Volatility']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Mini bar chart for recent volumes
            vol_fig = px.bar(filtered_df.tail(20), x='Date', y='Volume', title="Recent Volumes", template="plotly_dark")
            vol_fig.update_layout(height=300)
            st.plotly_chart(vol_fig, use_container_width=True)
    else:
        st.info("Please generate data first.")

# -----------------
# TAB 2: PREDICTIONS
# -----------------
with tabs[1]:
    st.subheader("Model Predictions Overlay (Out-of-Sample Test Set)")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Configuration")
        model_type = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "XGBoost", "SVR", "LSTM"])
        sentiment_config = st.radio("Sentiment Integration", ["With Sentiment", "Without Sentiment"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    suffix = "_sentiment" if sentiment_config == "With Sentiment" else ""
    pred_file = f"{selected_ticker}_{model_type.lower().replace(' ', '_')}{suffix}_predictions.csv"
    pred_path = os.path.join("models", pred_file)
    
    with col2:
        if os.path.exists(pred_path):
            pred_df = pd.read_csv(pred_path)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            
            # Interactive Predictions Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Actual'], name="Actual Close Price", line=dict(color='#1f77b4', width=2.5)))
            fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted'], name=f"Predicted ({model_type})", line=dict(color='#ff7f0e', width=2, dash='dash')))
            
            fig.update_layout(
                title=f"{selected_ticker} Actual vs. Predicted Price Comparison ({model_type} - {sentiment_config})",
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Zoomed in last 60 days
            fig_zoomed = go.Figure()
            zoomed_df = pred_df.tail(60)
            fig_zoomed.add_trace(go.Scatter(x=zoomed_df['Date'], y=zoomed_df['Actual'], name="Actual Price", line=dict(color='#1f77b4', width=2.5)))
            fig_zoomed.add_trace(go.Scatter(x=zoomed_df['Date'], y=zoomed_df['Predicted'], name="Predicted Price", line=dict(color='#e74c3c', width=2.5)))
            fig_zoomed.update_layout(
                title="Detailed Zoom: Last 60 Days",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig_zoomed, use_container_width=True)
        else:
            st.warning(f"Predictions file not found at {pred_path}. Please execute model training script.")

# -----------------
# TAB 3: RECOMMENDATIONS
# -----------------
with tabs[2]:
    st.subheader("Real-Time AI Buy/Hold/Sell Recommendations")
    
    # Selected model for recommendation
    rec_model = st.selectbox("Select Model for Recommendation Guidance", ["Linear Regression", "Random Forest", "XGBoost", "SVR", "LSTM"], key="rec_select")
    rec_sentiment = st.radio("Sentiment Option", ["With Sentiment", "Without Sentiment"], key="rec_sent_select")
    
    suffix = "_sentiment" if rec_sentiment == "With Sentiment" else ""
    rec_file = f"{selected_ticker}_{rec_model.lower().replace(' ', '_')}{suffix}_recommendations.csv"
    rec_path = os.path.join("models", rec_file)
    
    if os.path.exists(rec_path):
        rec_data = pd.read_csv(rec_path)
        last_rec = rec_data.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("### recommendation SIGNAL")
            sig = last_rec['Recommendation']
            if sig == 'Buy':
                st.markdown('<div class="rec-buy">BUY</div>', unsafe_allow_html=True)
            elif sig == 'Sell':
                st.markdown('<div class="rec-sell">SELL</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="rec-hold">HOLD</div>', unsafe_allow_html=True)
            st.markdown(f"**Predicted Return (Next Day):** {last_rec['predicted_return_pct']*100:+.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("### MODEL CONFIDENCE")
            conf = last_rec['model_confidence']
            # Dynamic representation
            st.markdown(f"<div style='font-size: 2.5rem; font-weight:800; color:#3d84ff;'>{conf:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"**Error Index (RMSE):** {last_rec['rolling_rmse']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("### MARKET RISK LEVEL")
            risk = last_rec['risk_level']
            color = "#00e676" if risk == "Low" else ("#ffea00" if risk == "Medium" else "#ff1744")
            st.markdown(f"<div style='font-size: 2.5rem; font-weight:800; color:{color};'>{risk.upper()}</div>", unsafe_allow_html=True)
            st.markdown(f"**Asset Volatility:** {last_rec['Volatility']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Display recommendation log
        st.subheader("Historical Recommendation Signals Logs (Recent 20 Days)")
        st.dataframe(rec_data.tail(20)[['Date', 'Close', 'Predicted', 'predicted_return_pct', 'model_confidence', 'risk_level', 'Recommendation']].rename(columns={
            'predicted_return_pct': 'Pred Return (%)',
            'model_confidence': 'Confidence',
            'risk_level': 'Risk Level'
        }).reset_index(drop=True), use_container_width=True)
    else:
        st.warning("Recommendation dataset not found. Run recommendation script first.")

# -----------------
# TAB 4: BACKTEST RESULTS
# -----------------
with tabs[3]:
    st.subheader("Backtesting: Algorithmic Strategy vs. Buy & Hold")
    
    backtest_model_type = st.selectbox("Select Strategy Core Model", ["Linear Regression", "Random Forest", "XGBoost", "SVR", "LSTM"], key="bt_select")
    backtest_sent = st.radio("Backtest Sentiment Suffix", ["With Sentiment", "Without Sentiment"], key="bt_sent_select")
    
    suffix = "_sentiment" if backtest_sent == "With Sentiment" else ""
    bt_file = f"{selected_ticker}_{backtest_model_type.lower().replace(' ', '_')}{suffix}_backtest.csv"
    bt_path = os.path.join("models", bt_file)
    
    if os.path.exists(bt_path):
        bt_df = pd.read_csv(bt_path)
        bt_df['Date'] = pd.to_datetime(bt_df['Date'])
        
        # Core Metrics
        last_bt = bt_df.iloc[-1]
        
        strategy_returns = bt_df['Strategy_Return'].values
        buyhold_returns = bt_df['actual_return_pct'].values
        
        # Calculate Sharpe
        def get_sharpe(ret):
            mean_ret = np.mean(ret)
            std_ret = np.std(ret)
            return (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0.0
            
        strat_sharpe = get_sharpe(strategy_returns)
        bh_sharpe = get_sharpe(buyhold_returns)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy Return", f"{last_bt['Cum_Strategy_Return']*100:.2f}%", delta=f"{(last_bt['Cum_Strategy_Return'] - last_bt['Cum_BuyHold_Return'])*100:+.2f}% vs B&H")
        col2.metric("Buy & Hold Return", f"{last_bt['Cum_BuyHold_Return']*100:.2f}%")
        col3.metric("Strategy Sharpe Ratio", f"{strat_sharpe:.2f}")
        col4.metric("Buy & Hold Sharpe", f"{bh_sharpe:.2f}")
        
        # Equity Curve Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt_df['Date'], y=bt_df['Cum_Strategy_Return']*100, name="AI Signals Strategy", line=dict(color='#2ca02c', width=2.5)))
        fig.add_trace(go.Scatter(x=bt_df['Date'], y=bt_df['Cum_BuyHold_Return']*100, name="Buy & Hold (Benchmark)", line=dict(color='#1f77b4', dash='dash', width=2)))
        
        fig.update_layout(
            title=f"Equity Curve: Strategy Cumulative Returns (%) - {selected_ticker} ({backtest_model_type})",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Backtest results not found for this configuration. Execute backtesting script.")

# -----------------
# TAB 5: SENTIMENT OVERVIEW
# -----------------
with tabs[4]:
    st.subheader("Financial News Sentiment Analytics & Text Corpus")
    
    # Load sentiment unscaled file
    sent_path = f"data/processed/{selected_ticker}_test_unscaled_sentiment.csv"
    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path)
        sent_df['Date'] = pd.to_datetime(sent_df['Date'])
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Sentiment Timeline Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sent_df['Date'], y=sent_df['sentiment_score'], name="Daily Sentiment Score", line=dict(color='rgba(255, 75, 77, 0.4)', width=1)))
            fig.add_trace(go.Scatter(x=sent_df['Date'], y=sent_df['sentiment_ma_3'], name="3-Day SMA Sentiment", line=dict(color='#ff4b4d', width=2)))
            
            fig.update_layout(
                title=f"{selected_ticker} Market Sentiment Timeline (VADER Lexicon)",
                template="plotly_dark",
                yaxis_title="Sentiment Index (-1 to 1)",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### Sentiment Metrics")
            mean_sent = sent_df['sentiment_score'].mean()
            st.markdown(f"**Mean Sentiment:** {mean_sent:+.3f}")
            
            pct_pos = (sent_df['sentiment_score'] > 0.05).mean() * 100
            pct_neg = (sent_df['sentiment_score'] < -0.05).mean() * 100
            pct_neut = 100 - pct_pos - pct_neg
            
            st.markdown(f"**Bullish Days:** {pct_pos:.1f}%")
            st.markdown(f"**Bearish Days:** {pct_neg:.1f}%")
            st.markdown(f"**Neutral Days:** {pct_neut:.1f}%")
            
            # Status
            market_tone = "Bullish 🐂" if mean_sent > 0.05 else ("Bearish 🐻" if mean_sent < -0.05 else "Neutral ⚖️")
            st.markdown(f"**Market Tone:** {market_tone}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Display simulated headlines sample
        st.subheader("Sample Text Headlines Scored by VADER Sentiment")
        # Generate some representative sample headlines for display matching the state
        sample_headlines = [
            {"Headline": f"{selected_ticker} releases innovative new product lineup to industry acclaim.", "VADER Compound": 0.5859, "Class": "Positive"},
            {"Headline": f"{selected_ticker} shares rise as quarterly revenue exceeds Wall Street targets.", "VADER Compound": 0.4404, "Class": "Positive"},
            {"Headline": f"EU antitrust regulators fine {selected_ticker} over developer policies.", "VADER Compound": -0.4939, "Class": "Negative"},
            {"Headline": f"{selected_ticker} schedules board meeting for next Wednesday.", "VADER Compound": 0.0000, "Class": "Neutral"},
            {"Headline": f"{selected_ticker} struggles with global microchip supply constraints.", "VADER Compound": -0.3182, "Class": "Negative"}
        ]
        st.table(pd.DataFrame(sample_headlines))
    else:
        st.info("Sentiment data not found. Run preprocessing and sentiment scripts.")

# -----------------
# TAB 6: MODEL COMPARISON
# -----------------
with tabs[5]:
    st.subheader("Consolidated Model performance Comparison")
    
    metrics_csv = "outputs/metrics.csv"
    if os.path.exists(metrics_csv):
        metrics_df = pd.read_csv(metrics_csv)
        
        # Display interactive metrics table
        st.dataframe(metrics_df.sort_values(['Ticker', 'RMSE']).reset_index(drop=True), use_container_width=True)
        
        # Bar charts of RMSE and R2
        col1, col2 = st.columns(2)
        
        ticker_metrics = metrics_df[metrics_df['Ticker'] == selected_ticker]
        
        with col1:
            fig_rmse = px.bar(
                ticker_metrics, 
                x='Model', 
                y='RMSE', 
                color='Features', 
                barmode='group',
                title=f"RMSE Comparison for {selected_ticker} (Lower is Better)",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.G10
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
            
        with col2:
            fig_r2 = px.bar(
                ticker_metrics, 
                x='Model', 
                y='R2', 
                color='Features', 
                barmode='group',
                title=f"R2 Comparison for {selected_ticker} (Higher is Better)",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_r2, use_container_width=True)
            
        # Display static shap/ablation charts as fallback/extra reference
        st.subheader("SHAP Feature Importance & Ablation Figures (Saved for Publication)")
        
        col_img1, col_img2 = st.columns(2)
        
        shap_glob_img = f"outputs/figures/{selected_ticker}_xgboost_shap_global.png"
        shap_loc_img = f"outputs/figures/{selected_ticker}_xgboost_shap_local.png"
        
        with col_img1:
            if os.path.exists(shap_glob_img):
                st.image(shap_glob_img, caption=f"SHAP Global Feature Importance ({selected_ticker} - XGBoost)", use_container_width=True)
            else:
                st.info("Run explainability.py to generate SHAP global charts.")
                
        with col_img2:
            if os.path.exists(shap_loc_img):
                st.image(shap_loc_img, caption=f"SHAP Local Explanation ({selected_ticker} - XGBoost)", use_container_width=True)
            else:
                st.info("Run explainability.py to generate SHAP local charts.")
    else:
        st.warning("Metrics comparison table not found. Run model training scripts first.")
