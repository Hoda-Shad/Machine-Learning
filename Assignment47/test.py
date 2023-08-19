import pandas as pd
from sklearn.model_selection import train_test_split
from perceptron import perceptron
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Surgical-deepnet.csv')
df.isnull().sum()
X = df.drop('complication', axis=1).copy().values
Y = df['complication'].copy().values
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

model = perceptron(learning_rate=0.001, input_length = X_train.shape[1])
model.fit(X_train, y_train, X_test, y_test, 64)
Y_pred = model.predict(X_test)
y_test = y_test.reshape(-1, 1)
print(confusion_matrix(Y_pred, y_test))
