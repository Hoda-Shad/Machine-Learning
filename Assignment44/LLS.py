import numpy as np 
from numpy.linalg import inv

class LLs: 
    def __init__(self):
        self.W = None
               
    def fit(self, X_train, y_train):
        self.W = inv(X_train.T @ X_train) @  X_train.T @ y_train   
        return self.W
        
    def predict(self, X_test):      
        y_pred = X_test @ self.W
        return y_pred


    def evaluate(self, X_test, y_test): 
        Y_pred = self.predict(X_test)
        acc = np.sum(Y_pred == y_test) / len(y_test)
        return acc


