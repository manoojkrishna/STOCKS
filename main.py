import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta
import plotly.graph_objs as go
import time
import logging
from textblob import TextBlob  # For sentiment analysis
from newsapi import NewsApiClient  # For fetching news articles

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress cmdstanpy logs
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Date range for data fetching
START = "2020-01-01"  # Reduced data range for faster loading
TODAY = date.today().strftime("%Y-%m-%d")

# Dictionary of full company names (Indian Stocks Only)
companies = {
    'RELIANCE.NS': 'Reliance Industries Ltd.',
    'TCS.NS': 'Tata Consultancy Services Ltd.',
    'HDFCBANK.NS': 'HDFC Bank Ltd.',
    'INFY.NS': 'Infosys Ltd.',
    'ICICIBANK.NS': 'ICICI Bank Ltd.',
    'HINDUNILVR.NS': 'Hindustan Unilever Ltd.',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel Ltd.',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank Ltd.',
    'LT.NS': 'Larsen & Toubro Ltd.',
    'AXISBANK.NS': 'Axis Bank Ltd.',
    'ASIANPAINT.NS': 'Asian Paints Ltd.',
    'MARUTI.NS': 'Maruti Suzuki India Ltd.',
    'NTPC.NS': 'NTPC Ltd.',
    'ONGC.NS': 'Oil and Natural Gas Corporation Ltd.',
    'ITC.NS': 'ITC Ltd.',
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries Ltd.',
    'WIPRO.NS': 'Wipro Ltd.',
    'POWERGRID.NS': 'Power Grid Corporation of India Ltd.',
    'NESTLEIND.NS': 'Nestle India Ltd.'
}

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='************************************')  # Replace with your NewsAPI key

# Fetch stock data with caching
@st.cache_data
def load_data(ticker, start=START, end=TODAY):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Prophet Forecast with caching
@st.cache_data
def prophet_forecast(data, periods=365):
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']

    model = Prophet(daily_seasonality=False)  # Disable daily seasonality for faster computation
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast, model

# Random Forest Prediction with caching
@st.cache_data
def random_forest_prediction(data):
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced n_estimators for faster training
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Generate future predictions
    future_dates = pd.date_range(start=data.index[-1], periods=30, freq='B')  # 30 future business days
    future_X = data[['Open', 'High', 'Low', 'Volume']].iloc[-30:]  # Use the last 30 days' data for future predictions
    future_predictions = model.predict(future_X)

    return predictions, mse, future_dates, future_predictions

# Fetch news articles and perform sentiment analysis
def fetch_news_and_sentiment(query):
    try:
        # Fetch news articles from the past 20 days
        from_date = (date.today() - timedelta(days=20)).strftime("%Y-%m-%d")
        to_date = TODAY
        news_articles = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='publishedAt',
            page_size=20  # Fetch top 20 articles
        )

        # Perform sentiment analysis on each article
        articles_with_sentiment = []
        for article in news_articles['articles']:
            title = article['title']
            source = article['source']['name']
            published_at = article['publishedAt']
            description = article['description']

            # Analyze sentiment using TextBlob
            blob = TextBlob(title)
            sentiment_score = blob.sentiment.polarity

            # Classify sentiment
            if sentiment_score > 0:
                sentiment = 'Positive'
            elif sentiment_score < 0:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'

            articles_with_sentiment.append({
                'Title': title,
                'Source': source,
                'Published At': published_at,
                'Sentiment': sentiment
            })

        return articles_with_sentiment

    except Exception as e:
        st.error(f"Error fetching news articles: {e}")
        return []

# Live Stock Price with Enhanced Candlestick Chart (using yfinance)
def live_stock_price_tab(selected_stock, company_name):
    st.subheader(f'Live Candlestick Chart for {company_name} ({selected_stock})')

    live_data = {'Time': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Color': []}
    st.write("Fetching live stock prices...")
    live_chart = st.empty()

    try:
        for _ in range(60):  # Fetch live prices for 60 seconds
            ticker = yf.Ticker(selected_stock)
            history = ticker.history(period="1d", interval="1m")
            if not history.empty:
                live_price = history['Close'].iloc[-1]
                timestamp = pd.Timestamp.now()

                if len(live_data['Time']) > 0:
                    prev_close = live_data['Close'][-1]
                else:
                    prev_close = live_price

                # Simulate OHLC values
                open_price = prev_close
                close_price = live_price
                high_price = max(prev_close, live_price)
                low_price = min(prev_close, live_price)

                # Determine color: Green (Increase), Red (Decrease), Blue (Stable)
                if close_price > open_price:
                    color = 'green'  # Price Increased
                elif close_price < open_price:
                    color = 'red'  # Price Decreased
                else:
                    color = 'blue'  # No Change (Stable)

                # Store the values
                live_data['Time'].append(timestamp)
                live_data['Open'].append(open_price)
                live_data['High'].append(high_price)
                live_data['Low'].append(low_price)
                live_data['Close'].append(close_price)
                live_data['Color'].append(color)

                df_live = pd.DataFrame(live_data)

                # Create Candlestick Chart with custom colors
                fig = go.Figure()

                for i in range(len(df_live)):
                    fig.add_trace(go.Candlestick(
                        x=[df_live['Time'][i]],
                        open=[df_live['Open'][i]],
                        high=[df_live['High'][i]],
                        low=[df_live['Low'][i]],
                        close=[df_live['Close'][i]],
                        increasing_line_color=df_live['Color'][i],
                        decreasing_line_color=df_live['Color'][i],
                    ))

                fig.update_layout(
                    title=f'Live Candlestick Chart for {company_name}',
                    xaxis_title='Time',
                    yaxis_title='Price (INR)',
                    xaxis_rangeslider_visible=False,
                    template='plotly_dark'
                )

                # Update Chart
                live_chart.plotly_chart(fig, use_container_width=True)
                time.sleep(2)  # Update every 2 seconds
            else:
                st.warning("No live data available for the selected stock.")
                break

    except Exception as e:
        st.error(f"Error fetching live stock price: {e}")

# Main Streamlit App
def main():
    st.title('Advanced Stock Prediction App')

    selected_stock = st.selectbox('Select a Stock', list(companies.keys()), format_func=lambda x: companies[x])
    data = load_data(selected_stock)

    if data is None or data.empty:
        st.error("Could not load stock data")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        'Stock Overview',
        'Live Stock Price',
        'Prophet Forecast',
        'Random Forest Prediction',
        'Sentiment Analysis',
        'About'
    ])

    with tab1:
        st.subheader(f'Stock Overview for {companies[selected_stock]} ({selected_stock})')
        st.write("Recent Stock Data:")
        st.dataframe(data.tail())

        # Plot historical stock data
        st.write("Historical Stock Price:")
        fig, ax = plt.subplots()
        data['Close'].plot(ax=ax)
        ax.set_title(f'{selected_stock} Stock Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (INR)')
        st.pyplot(fig)

    with tab2:
        live_stock_price_tab(selected_stock, companies[selected_stock])

    with tab3:
        st.subheader(f'Prophet Forecast for {companies[selected_stock]}')
        try:
            forecast, model = prophet_forecast(data)

            # Plot forecast
            st.write("Forecast Data:")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            st.write("Forecast Plot:")
            fig = model.plot(forecast)
            st.pyplot(fig)

            st.write("Forecast Components:")
            fig_components = model.plot_components(forecast)
            st.pyplot(fig_components)
        except Exception as e:
            st.error(f"Prophet Forecast Error: {e}")

    with tab4:
        st.subheader(f'Random Forest Prediction for {companies[selected_stock]}')
        try:
            predictions, mse, future_dates, future_predictions = random_forest_prediction(data)

            st.write("Random Forest Predictions vs Actual:")
            fig, ax = plt.subplots()
            ax.plot(data.index[-len(predictions):], predictions, label='Predicted')
            ax.plot(data.index[-len(predictions):], data['Close'].iloc[-len(predictions):], label='Actual')
            ax.set_title('Random Forest Predictions vs Actual')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (INR)')
            ax.legend()
            st.pyplot(fig)

            st.write(f"Mean Squared Error: {mse}")

            # Display future predictions in a table
            st.write("Future Predicted Values:")
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Close Price': future_predictions
            })
            st.dataframe(future_df)

        except Exception as e:
            st.error(f"Random Forest Prediction Error: {e}")

    with tab5:
        st.subheader(f'Sentiment Analysis for {companies[selected_stock]}')
        query = companies[selected_stock]  # Use the company name as the query
        articles_with_sentiment = fetch_news_and_sentiment(query)

        if articles_with_sentiment:
            st.write("News Articles with Sentiment Analysis:")
            sentiment_df = pd.DataFrame(articles_with_sentiment)
            st.dataframe(sentiment_df)

            # Display sentiment distribution
            sentiment_counts = sentiment_df['Sentiment'].value_counts()
            st.write("Sentiment Distribution:")
            st.bar_chart(sentiment_counts)
        else:
            st.warning("No news articles found for the selected stock.")

    with tab6:
        st.subheader('About')
        st.write("""
        This app provides stock price predictions using:
        - **Prophet**: A time series forecasting tool by Facebook.
        - **Random Forest**: A machine learning model for regression.
        - **Sentiment Analysis**: Analysis of news articles using TextBlob.
        """)
        st.write("Developed using Python, Streamlit, and Yahoo Finance.")

if __name__ == '__main__':
    main()
