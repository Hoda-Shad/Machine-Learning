import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from perceptron_Boston_1 import perceptron

data = pd.read_csv('Dataset/Boston.csv')
print(data['indus'])
X = np.array(data[['indus', 'rad']])
print(X.shape)
Y = np.array(data['tax'])
Y = Y.reshape(-1, 1)

# print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y)

epochs = 20
w = np.random.rand(1, 2)

print(w.shape)
b = np.random.rand(1, 1)
prcptrn = perceptron(w, b)
prcptrn.fit(X_train, y_train, epochs)
Y_pred = prcptrn.predict(X_test)
loss = prcptrn.evaluate(X_test, y_test)
