'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.metrics import mean_squared_error

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url, usecols=[1])
data = data.values.astype('float32')

#Normalize the dataset

scaler = MinMaxScaler(feature_range = (0,1))
data_scaled =  scaler.fit_transform(data)

train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled[0:train_size], data_scaled[train_size:]

#COnvert to supervised learning problem

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range (len(dataset) - look_back - 1):
        X.append(dataset[i: (i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 5
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 1))

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


model_lstm = Sequential()
model_lstm.add(LSTM(50, input_shape=(look_back, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X_train, y_train, epochs = 100, batch_size =1, verbose = 0)

#Build GRU model

model_gru = Sequential()
model_gru.add(GRU(50, input_shape=(look_back, 1)))
model_gru.add(Dense(1))
model_gru.compile(loss='mean_squared_error', optimizer = 'adam')
model_gru.fit(X_train, y_train, epochs = 100, batch_size = 1, verbose = 0)

#MAke prediction
lstm_predictions = model_lstm.predict(X_test)
gru_predictions = model_gru.predict(X_test)

#Inverse transform prediction and actual values

lstm_predictions = scaler.inverse_transform(lstm_predictions)
gru_predictions = scaler.inverse_transform(gru_predictions)
y_test_inverse = scaler.inverse_transform([y_test])

#Calculate rmse for both models

lstm_rmse = np.sqrt(mean_squared_error(y_test_inverse[0], lstm_predictions[:, 0]))
gru_rmse = np.sqrt(mean_squared_error(y_test_inverse[0], gru_predictions[:, 0]))

#plot the predictions and actual
plt.figure(figsize=(12, 6))
plt.plot(y_test_inverse[0], label = 'Actual Data')
plt.plot(lstm_predictions[:, 0], label = 'LSTM Predcitions')
plt.plot(gru_predictions[:, 0], label = 'GRU Predcitions')
plt.title('LSTM Vs GRU')
plt.xlabel('Time steps')
plt.ylabel('NUmber of passengers')
plt.legend()
plt.show()

lstm_rmse, gru_rmse'''