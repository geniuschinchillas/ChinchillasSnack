# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:22:49 2019

@author: holya
"""

import tensorflow as tf
print(tf.__version__)

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

print(dataset_train.shape)

tf.random.set_seed(13)

open_data = dataset_train['Open']
open_data.index = dataset_train['Date']

open_data.plot(subplots=True)

open_data = open_data.values

open_data_mean = open_data.mean()
open_data_std =  open_data.std()

open_data = (open_data - open_data_mean)/open_data_std

X_train_open = []
y_train_open = []

for i in range(90, 1258):
    #print(open_data[i])
    X_train_open.append(open_data[i-90:i])
    y_train_open.append(open_data[i])

X_train_open, y_train_open = np.array(X_train_open), np.array(y_train_open)

print ('Single window of past history')
print (X_train_open[0])
print ('\n Target temperature to predict')
print (y_train_open[0])

def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

show_plot([X_train_open[0], y_train_open[0]], 0, 'Sample Example')

def baseline(history):
  return np.mean(history)

show_plot([X_train_open[0], y_train_open[0], baseline(X_train_open[0])], 0,
           'Baseline Prediction Example')

BATCH_SIZE = 32
BUFFER_SIZE = 10000

X_train_open = np.reshape(X_train_open, (X_train_open.shape[0], X_train_open.shape[1],1))

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=(90,1), return_sequences = True),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

EVALUATION_INTERVAL = 8
EPOCHS = 50

simple_lstm_model.fit(X_train_open, y_train_open, epochs=EPOCHS, batch_size=32)


