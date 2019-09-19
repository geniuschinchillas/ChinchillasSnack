# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:45:47 2019

@author: garli
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []

for i in range(60, 1258):
    
    
