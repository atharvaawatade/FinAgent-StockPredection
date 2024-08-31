import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
import json
import warnings
import os
import requests

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load RapidAPI credentials from environment variable
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# Expanded company name to stock symbol mapping
company_symbol_mapping = {
    "Netflix": "NFLX",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    # ... (rest of the mapping)
}

# Function to fetch stock data from RapidAPI
def fetch_stock_data(symbol, time_series_type, interval=None):
    url = "https://alpha-vantage.p.rapidapi.com/query"
    
    querystring = {
        "function": f"TIME_SERIES_{time_series_type.upper()}",
        "symbol": symbol,
        "outputsize": "compact",
        "datatype": "json"
    }
    
    if time_series_type == "Intraday":
        querystring["interval"] = interval
    
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers, params=querystring)
    
    if response.status_code != 200:
        st.error("Failed to fetch data from RapidAPI. Please try again later.")
        return None
    
    data_json = response.json()
    
    if "Time Series" not in data_json:
        st.error(f"API Error: {data_json.get('Error Message', 'Unknown error')}")
        return None
    
    time_series_key = f"Time Series ({interval})" if time_series_type == "Intraday" else f"{time_series_type} Time Series"
    time_series = data_json.get(time_series_key)
    return time_series

# Function to convert fetched data into a DataFrame
def convert_to_dataframe(time_series):
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    df = df.astype(float)
    return df

# Function to train a model and predict stock trend
def predict_trend(df, period, freq):
    df_prophet = df.reset_index().rename(columns={"index": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=period, freq=freq.lower())
    forecast = model.predict(future)

    fig = model.plot(forecast)
    st.pyplot(fig)

    forecast['trend_diff'] = forecast['yhat'].diff()
    last_trend = forecast['trend_diff'].iloc[-1]
    return "Up" if last_trend > 0 else "Down" if last_trend < 0 else "Neutral"

# Function to plot the stock data
def plot_stock_data(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=f'Close Price of {symbol}', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines', name=f'Open Price of {symbol}', line=dict(color='green')))
    fig.update_layout(
        title=f'Stock Price of {symbol} Over Time',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x',
        template='plotly_dark'
    )
    st.plotly_chart(fig)

# Streamlit app
def main():
    st.title("FinAgent Stock Prediction")
    st.markdown("""
    <style>
    .reportview-container {
        background-color: #f5f5f5;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    search_term = st.text_input("Search for a Company", "")
    
    if search_term:
        filtered_companies = {name: symbol for name, symbol in company_symbol_mapping.items() if search_term.lower() in name.lower()}
        
        if filtered_companies:
            for company_name, stock_symbol in filtered_companies.items():
                st.markdown(f"### {company_name} ({stock_symbol})")
                
                time_series_type = st.selectbox("Select Time Series Type", ["Intraday", "Weekly", "Monthly"], key=company_name)
                
                interval = st.selectbox("Select Interval", ["1min", "5min", "15min", "30min", "60min"], key=f'{company_name}_interval') if time_series_type == "Intraday" else None
                
                if st.button(f"Fetch Data for {company_name}"):
                    with st.spinner(f"Fetching data for {company_name}..."):
                        time_series = fetch_stock_data(stock_symbol, time_series_type, interval)
                    
                    if time_series:
                        df = convert_to_dataframe(time_series)
                        st.write(f"Showing {time_series_type.lower()} data for: **{company_name} ({stock_symbol})**")
                        st.dataframe(df.head())
                        
                        plot_stock_data(df, company_name)
                        
                        period_freq_mapping = {'Intraday': ('1', 'H'), 'Weekly': ('1', 'W'), 'Monthly': ('1', 'M')}
                        period, freq = period_freq_mapping.get(time_series_type, ('1', 'D'))
                        trend = predict_trend(df, int(period), freq)
                        st.write(f"Trend Prediction for {company_name}: {trend}")

if __name__ == "__main__":
    main()
