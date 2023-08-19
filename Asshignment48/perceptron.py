import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class perceptron:

    def __init__(self, learning_rate, input_length):
        # input_length is the number of features (here 24 )
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_length)
        self.bias = np.random.rand(1)

    def activation(self, x, function):
        if function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif function == "relu":
            return np.maximum(0, x)
        elif function == "tanh":
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        elif function == "linear":
            return x
        elif function == "binarystep":
            return np.heaviside(x, 1)

    def fit(self, X_train, y_train, X_test, y_test, epochs):

        acc_train = []
        acc_test = []
        loss_train = []
        loss_test = []
        for epoch in tqdm(range(epochs)):
            for x, y in zip(X_train, y_train):
                # forwarding
                y_pred = x @ self.weights + self.bias
                y_pred = self.activation(y_pred, "sigmoid")

                # backpropagation
                error = y - y_pred

                # updating
                self.weights = self.weights + self.learning_rate * x * error
                self.bias = self.bias + self.learning_rate * error

            acc_train.append(self.evaluate(X_train, y_train)[1])
            acc_test.append(self.evaluate(X_test, y_test)[1])
            loss_train.append(self.evaluate(X_train, y_train)[0])
            loss_test.append(self.evaluate(X_test, y_test)[0])


    def calculate_loss(self, X_test, y_test, metric):
        y_pred = self.predict(X_test)
        if metric == 'mae':
            return np.sum(np.abs(y_test - y_pred))
        else:
            raise Exception("Unknown Metric")

    def calculate_accuracy(self, X, y):
        y_pred = self.predict(X)
        # y_pred = np.where(y_pred > 0, 1, 0)
        y_pred = y_pred.reshape(-1,1)
        acc = np.mean(y_pred == y)
        return acc

    def predict(self, X_test):
        Y_pred = []
        for x in X_test:
            y_pred = x @ self.weights + self.bias
            y_pred = self.activation(y_pred, 'sigmoid')
            y_pred = np.where(0.5<y_pred, 1, 0)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def evaluate(self, X_test, y_test):
        loss = self.calculate_loss(X_test, y_test, 'mae')
        accuracy = self.calculate_accuracy(X_test, y_test)
        return loss, accuracy
