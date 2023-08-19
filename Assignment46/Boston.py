import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from perceptron_Boston_1 import perceptron
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Dataset/Boston.csv')
cor = data.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

target = abs(data['medv'])
relevant_features = target[target>0.5]
# print(relevant_features)

X = np.array(data[['lstat', 'ptratio']])
Y = np.array(data['medv'])
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
