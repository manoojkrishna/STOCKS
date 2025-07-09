# 📈 Advanced Stock Market Prediction App (India)

This is a **Streamlit web application** for analyzing and forecasting stock prices of top Indian companies. It uses **Prophet** for time series forecasting, **Random Forest** for machine learning regression, **TextBlob** for sentiment analysis, and **NewsAPI** for fetching real-time news.

---

## 🚀 Features

### 📊 Stock Overview
- Displays historical stock data from Yahoo Finance.
- Interactive chart of stock price over time.

### 🔴 Live Stock Price
- Real-time candlestick chart updated every 2 seconds.
- Custom colored OHLC visualization (Green for up, Red for down, Blue for no change).

### 🔮 Prophet Forecast
- Forecast future stock prices using Facebook's Prophet model.
- Shows forecast plot and decomposed components.

### 🌲 Random Forest Prediction
- Predict closing stock prices using machine learning (Random Forest).
- Shows prediction vs actual chart, MSE, and 30-day future predictions.

### 📰 Sentiment Analysis
- Fetches latest news articles from NewsAPI.
- Performs sentiment analysis (Positive / Negative / Neutral) using TextBlob.
- Displays sentiment distribution chart.

### ℹ️ About
- App information and methodology.

---

## 🏢 Supported Indian Companies

- Reliance Industries Ltd. (`RELIANCE.NS`)
- Tata Consultancy Services (`TCS.NS`)
- HDFC Bank (`HDFCBANK.NS`)
- Infosys (`INFY.NS`)
- ICICI Bank (`ICICIBANK.NS`)
- Hindustan Unilever (`HINDUNILVR.NS`)
- State Bank of India (`SBIN.NS`)
- Bharti Airtel Ltd. (`BHARTIARTL.NS`)
- Larsen & Toubro (`LT.NS`)
- Kotak Mahindra Bank (`KOTAKBANK.NS`)
- Axis Bank (`AXISBANK.NS`)
- Asian Paints (`ASIANPAINT.NS`)
- Maruti Suzuki (`MARUTI.NS`)
- NTPC (`NTPC.NS`)
- ONGC (`ONGC.NS`)
- ITC (`ITC.NS`)
- Sun Pharma (`SUNPHARMA.NS`)
- Wipro (`WIPRO.NS`)
- Power Grid Corp (`POWERGRID.NS`)
- Nestle India (`NESTLEIND.NS`)

---

## 🧠 Technologies Used

| Tool | Purpose |
|------|---------|
| `Streamlit` | Interactive Web UI |
| `yfinance` | Stock price data |
| `Prophet` | Time series forecasting |
| `RandomForestRegressor` | ML prediction |
| `TextBlob` | Sentiment analysis |
| `NewsAPI` | Real-time news |
| `Plotly` & `Matplotlib` | Data visualization |

---

## 📦 Required Python Packages

Install all necessary packages using the following command:

```bash
pip install streamlit yfinance pandas numpy matplotlib prophet scikit-learn plotly textblob newsapi-python
