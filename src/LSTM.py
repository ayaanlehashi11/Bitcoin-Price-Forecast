from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score , r2_score
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
mod = load_model("model/gru_model.h5")
data = pd.read_csv("data/BTC-USD.csv")
x = data[["Close", "High", "Low"]]
y = data["Open"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(xtrain)
X_test_scaled = scaler.transform(xtest)
data_open = data.filter(['open'])
dataset = data_open.values
mean = dataset.mean(axis=0)
std = dataset.std(axis=0)
dataset -= mean
dataset /= std
count = 0
predictions = mod.predict(xtrain)
print(r2_score(ytrain , predictions))
for i in range(1 , 26908):
    if predictions[i - 1] < predictions[i]:
        print("1")
    else:
        print("0")
