import numpy as np
from tqdm import tqdm 

class perceptron:
    
    def __init__(self, learning_rate, input_length):
        #input_lenght is the number of features (here 24 )
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_length)
        self.bias = np.random.rand(1)


    def activation(self, x, function):
        if function == "sigmoid" :
            return 1 / (1 + np.exp(-x))
        elif function == "relu" :
            return np.maximum(0,x)


    def fit(self, X_train, y_train, epochs):
        for epoch in tqdm(range(epochs)): 
            for x,y in zip(X_train , y_train):
                # forwarding
                y_pred = x @ self.weights + self.bias
                y_pred = self.activation(y_pred , "sigmoid")
                #backpropagation 
                error = y - y_pred
                #updating
                self.weights +=  self.learning_rate * x * error
                self.bias = self.bias + self.learning_rate * x 
            
                
    def calculate_loss(self, X_test, y_test, metric):
        y_pred = self.predict(X_test)
        if metric == 'mse' :
            return np.mean(np.square(y_test - y_pred))
        else:
            raise Exception("Unknown Metric")

    def calculate_accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred = np.where(y_pred > 0.5 , 1)
        acc = np.sum(y_pred == y_test) / len(y_test)
        return acc
            
    def predict(self, X_test): 
        Y_pred = []
        for x in X_test : 
            y_pred = x * self.weights + self.bias 
            y_pred = self.activation(y_pred, 'sigmoid')
            y_pred = np.where( y_pred > 0.5 , 1)
            Y_pred.append(y_pred)
            return Y_pred

    def evaluate(self, X_test, y_test):
        loss = self.calculate_loss (X_test, y_test, 'mse')
        y_pred = self.predict(X_test)
        y_pred = np.where(y_pred > 0.5)
        accuracy = self.calculate_accuracy(X_test,y_test)
        return loss , accuracy 
        