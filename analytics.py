import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import requests
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, GRU, Dropout
from tensorflow.keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers

def plot_bitcoin_data(coin,type):
    # Fetch Bitcoin data
    desired_date = datetime.datetime(2023, 1, 1)
    start_date = desired_date - datetime.timedelta(days=30)
    start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    url = f"https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_{coin}_USD/history?period_id=1DAY&time_start={start_date_str}&time_end=2023-03-31T00:00:00"
    headers = {"X-CoinAPI-Key": "4AEEDBE5-C3F5-42C8-ABC4-CDD4B54FE078"}  # Replace with your API key
    response = requests.get(url, headers=headers)
    print(url)

    # Check if the response is successful
    if response.status_code != 200 or not response.content:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return

    # Convert data to DataFrame
    data = response.json()
    df = pd.DataFrame(data)

    # Convert 'time_period_start' column to datetime and extract date
    df['date'] = pd.to_datetime(df['time_period_start'])

    # Drop unnecessary columns
    df.drop(['time_period_start', 'time_period_end', 'time_open', 'time_close'], axis=1, inplace=True)
    print("HI")
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['price_close'], color='blue', linestyle='-')
    plt.title(f'{coin} Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if (type=="history"):
        plt.savefig(f'static/{coin}_history.png')
    plt.close()

    # Prepare data for LSTM
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']])

# Create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

# Choose sequence length
    sequence_length = 10

# Create sequences
    X, y = create_sequences(scaled_data, sequence_length)
         # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dropout, Dense
    from tensorflow.keras import regularizers
    
    n_steps = X_train.shape[1]
    
    model = Sequential()
    model.add(GRU(1000, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), input_shape=(n_steps, X_train.shape[2])))
    model.add(Dropout(0.05))
    model.add(GRU(1000, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    model.add(Dropout(0.05))
    model.add(Dense(X_train.shape[2]))

    model.compile(optimizer='adam', loss='mse')



    model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.1)
       # Get the last sequence from the training data
    last_sequence = X_train[-1:]


    predicted = []
    current_sequence = last_sequence.copy()

    for _ in range(30):
        prediction = model.predict(current_sequence)
        predicted.append(prediction)
        current_sequence = np.append(current_sequence[:,1:,:], [prediction], axis=1)


    predicted = np.array(predicted)
    predicted = scaler.inverse_transform(predicted.reshape(-1, 5))

    predicted_df = pd.DataFrame(predicted, columns=['price_open', 'price_high', 'price_low', 'price_close', 'volume_traded'])

    # Calculate Bollinger Bands
    df['SMA_20'] = df['price_close'].rolling(window=20).mean()
    df['std_dev'] = df['price_close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (2 * df['std_dev'])
    df['Lower_Band'] = df['SMA_20'] - (2 * df['std_dev'])

# Calculate MACD
    df['EMA_12'] = df['price_close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['price_close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

# Plot the data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot Bollinger Bands
    ax1.plot(df.index, df['price_close'], label='Close Price')
    ax1.plot(df.index, df['Upper_Band'], label='Upper Band')
    ax1.plot(df.index, df['Lower_Band'], label='Lower Band')
    ax1.set_title('Bitcoin Price and Bollinger Bands')
    ax1.legend()
    

# Plot MACD
    ax2.plot(df.index, df['MACD'], label='MACD')
    ax2.plot(df.index, df['Signal_Line'], label='Signal Line')
    ax2.bar(df.index, df['MACD_Histogram'], label='MACD Histogram')
    ax2.set_title('Bitcoin MACD')
    ax2.legend()


    if (type=="macd"):
         plt.savefig(f'static/{coin}_macd.png')
    plt.close()

   

    def candlestick_ohlc(ax, quotes, width=0.6, colorup='k', colordown='r', alpha=1.0):
        OFFSET = width / 2
        opens = quotes[:, 0]
        highs = quotes[:, 1]
        lows = quotes[:, 2]
        closes = quotes[:, 3]
    
        dirs = np.where(closes >= opens, np.ones_like(closes), -np.ones_like(closes))
        ranges = np.dstack((opens - lows, highs - closes))[0]
    
        ax.bar(np.arange(len(quotes)), dirs * ranges[:, 0], width, bottom=lows, color=colorup, label='Bullish')
        ax.bar(np.arange(len(quotes)), dirs * ranges[:, 1], width, bottom=closes, color=colordown, alpha=alpha, label='Bearish')
        ax.plot(np.arange(len(quotes)), closes, color='b', label='Close Price')

# Assuming 'data' is your OHLC data
    df = pd.DataFrame(data)

# Convert OHLC data to float
    df[['price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']] = df[['price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']].astype(float)

# Convert the 'time_period_start' column to datetime
    df['time_period_start'] = pd.to_datetime(df['time_period_start'])

# Create a candlestick chart
    fig, ax = plt.subplots(figsize=(12, 6))
    candlestick_ohlc(ax, df[['price_open', 'price_high', 'price_low', 'price_close']].values, width=0.6, colorup='g', colordown='r', alpha=0.8)
    
# Format the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.set_xticks(np.arange(0, len(df), 1))
    ax.set_xticklabels(df['time_period_start'].dt.strftime('%Y-%m-%d'), rotation=90)

# Add legend
    plt.legend()

# Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Candlestick Chart')

    if (type=="candlestick"):
         plt.savefig(f'static/{coin}_candlestick.png')
    plt.close()
    



