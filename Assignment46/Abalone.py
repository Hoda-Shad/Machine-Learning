import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from perceptron_oop import perceptron

data = pd.read_csv('Dataset/abalone.csv')
data["Sex"] = data["Sex"].replace (["F", "M" , "I"], [0,1,2])
print(data.corr())
X = np.array(data["Length"])
Y = np.array(data["Height"])
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y)
print(X_train.shape)
epochs = 20
w = np.random.rand(1,1)
b = np.random.rand(1,1)
prcptrn = perceptron(w,b)
prcptrn.fit(X_train, y_train, epochs)
Y_pred = prcptrn.predict(X_test)
loss = prcptrn.evaluate(X_test, y_test)
