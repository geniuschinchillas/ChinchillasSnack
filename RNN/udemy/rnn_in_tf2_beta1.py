# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:23:13 2019

@author: holya
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os

print(tf.__version__)

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

print(dataset_train.shape)
print(dataset_test.shape)

training_set = dataset_train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train_open = []
y_train_open = []

for i in range(60, 1258):
    #print(open_data[i])
    X_train_open.append(training_set_scaled[i-60:i])
    y_train_open.append(training_set_scaled[i])
    
X_train_open, y_train_open = np.array(X_train_open), np.array(y_train_open)

X_train_open = np.reshape(X_train_open, (X_train_open.shape[0], X_train_open.shape[1],1))

BATCH_SIZE = 32
EPOCHS = 50

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=(60,1), return_sequences = True),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

simple_lstm_model.fit(X_train_open, y_train_open, epochs=EPOCHS, batch_size=BATCH_SIZE)

real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = simple_lstm_model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price[0])

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()