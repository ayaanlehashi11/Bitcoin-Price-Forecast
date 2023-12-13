from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import math
InputNet = pd.read_csv("data/EURUSD.csv")
InputIndicatorNet = pd.read_csv("data/EURUSDIndicatorMin.csv")
# data_open = InputNet.filter(['open'])
# dataset = data_open.values
x = InputNet[['close', 'high', 'low']]
y = InputNet['open']
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)
xtrain , xtest , ytrain , ytest = train_test_split(x,y, test_size=0.2 , random_state=0)
model = Sequential()
model.add(LSTM(100, return_sequences = True, input_shape=(xtrain.shape[1],1)))
model.add(LSTM(70, return_sequences = False))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
print(model.summary())
model.fit(xtrain , ytrain, epochs=100, batch_size=10,verbose=2,validation_split=0.3)
model.save('lstm_model.h5')





