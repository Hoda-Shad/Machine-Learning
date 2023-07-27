import numpy as np
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap



class KNN:
    def __init__(self,k):
        self.k = k 
        
    def fit(self,X,Y):
        self.X_train = X
        self.Y_train = Y
        
    def predict(self,X):
        Y = []
        for x in X: 
            distances = []
            for x_train in self.X_train:
                d = self.euclidean_distance(x,x_train) 
                distances.append(d)
            nearest_neighbors = np.argsort(distances)[0:self.k]
            y = np.argmax(np.bincount(self.Y_train[nearest_neighbors]))
            Y.append(y)
        return Y
        
    def evaluate(self, X , Y):
        Y_pred = self.predict(X)
        acc = np.sum(Y_pred==Y) / len(Y)
        return acc
        
    def euclidean_distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))