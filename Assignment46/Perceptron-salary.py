import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from perceptron_oop import perceptron

# Data
x, y, coef = datasets.make_regression(n_samples=100,#number of samples
                                      n_features=1,#number of features
                                      n_informative=1,#number of useful features
                                      noise=10,#bias and standard deviation of the guassian noise
                                      coef=True,#true coefficient used to generated the data
                                      random_state=0) #set for same data points for each run

x = np.interp(x, (x.min(), x.max()), (0, 20))
y = np.interp(y, (y.min(), y.max()), (20000, 150000))
plt.ion() #interactive plot on
plt.plot(x,y,'.',label='training data')
plt.xlabel('Years of experience');plt.ylabel('Salary $')
plt.title('Experience Vs. Salary')

#Train
X_train , X_test, y_train , y_test = train_test_split(x, y, test_size = 0.2)
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
epochs = 20
w = np.random.rand(1,1)
b = np.random.rand(1,1)
prcptrn = perceptron(w,b)
prcptrn.fit(X_train, y_train, epochs)
Y_pred = prcptrn.predict(X_test)
loss = prcptrn.evaluate(X_test, y_test)
