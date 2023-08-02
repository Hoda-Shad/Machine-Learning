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
        y_pred = self.predict(X_test)
        # loss = np.mean (np.sum (np.abs (y_test - y_pred)))
        mae = np.absolute(np.subtract(y_test, y_pred)).mean()
        return mae


