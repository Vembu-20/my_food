'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf

# Download data
stock_data = yf.download('AAPL', start='2012-01-01', end='2022-01-01')
df = stock_data[['Close']]

# Rolling mean smoothing
df['Close'] = df['Close'].rolling(window=5).mean()
df.dropna(inplace=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Dataset creation function
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = y[:train_size], y[train_size:]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, Y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, Y_test),
                    callbacks=[early_stop],
                    verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train_inv = scaler.inverse_transform(Y_train.reshape(-1, 1))
Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

train_rmse = np.sqrt(mean_squared_error(Y_train_inv, train_predict))
test_rmse = np.sqrt(mean_squared_error(Y_test_inv, test_predict))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Future predictions
n_future = 100
last_sequence = scaled_data[-time_step:]
predictions = []

for _ in range(n_future):
    next_prediction = model.predict(last_sequence.reshape(1, time_step, 1))
    predictions.append(next_prediction[0, 0])
    last_sequence = np.append(last_sequence[1:], next_prediction, axis=0)

predictions = np.array(predictions)
future_predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Actual', color='blue')
plt.plot(df.index[time_step:time_step + len(train_predict)], train_predict, color='green', label='Train Predictions')
plt.plot(df.index[train_size + time_step + 1:train_size + time_step + 1 + len(test_predict)], test_predict, color='red', label='Test Predictions')

future_dates = pd.date_range(df.index[-1], periods=n_future + 1, freq='D')[1:]
plt.plot(future_dates, future_predictions, color='orange', label='Future Predictions')

plt.legend()
plt.title('LSTM Prediction Vs Actual')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()'''