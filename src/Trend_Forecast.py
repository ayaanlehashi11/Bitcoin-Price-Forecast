import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
# from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.metrics import MeanAbsolutePercentageError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , r2_score ,  mean_absolute_error
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/EURUSD.csv')
# print(data.shape)
# print(data.head())
x = data[["close", "high", "low"]]
y = data["open"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(xtrain)
X_test_scaled = scaler.transform(xtest)
"""model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
data2 = pd.DataFrame(data={"Predicted Rate": ypred.flatten()})
print(data2.head())
print(data2.shape)"""
model = RandomForestRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtrain)
h_model = load_model("model/gru_model.h5")
pred = h_model.predict(xtest)
print(mean_squared_error(ytrain, ypred))
acc = r2_score(ytest , pred)
accuracy =  r2_score(ytrain , ypred)
print(f"optimized accuracy = {accuracy*100} % ")
print(f"optimized accuracy = {acc*100} %")
"""
data2 = pd.DataFrame(data={"Predicted Rate": ypred.flatten()})
print(data2.shape)
print(data2["Predicted Rate"])"""

"""import pickle

pickle.dump(model, open('rfr_model.sav', 'wb'))

loaded_model = pickle.load(open('rfr_model.sav', 'rb'))
xpred = loaded_model.predict(xtest)
acc = r2_score(ytest , xpred)
print(f"the accuracy in the test dataset is {acc}")
# print(data.shape)
# print(data.head())
# data = data.reshape(len(data), 1, data.shape[1])
# or
# data = np.expand_dims(data, 1)
"""
"""
model = Sequential()
model.add(LSTM(16,return_sequences=True,activation='relu' , input_shape=(xtrain.shape[1:])))
model.add(Dropout(0.3))
model.add(LSTM(16,return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(optimizer='adam',
                loss='mse',
                metrics=['mae','mse'])
print(model.summary())
model.fit(x=xtrain, y=ytrain, epochs=25,batch_size=500,validation_data=(xtest,ytest))

model.save('model.h5')

"""