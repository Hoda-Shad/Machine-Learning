import numpy as np
from tqdm import tqdm


class Perceptron:
    def __init__(self, learning_rate, input_length, epochs):
        self.lr = learning_rate
        self.w = np.random.rand(input_length)
        self.b = np.random.rand(1)
        self.epochs = epochs

    def activation(self, x, function='relu'):
        if function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif function == 'relu':
            return np.maximum(0, x)
        elif function == 'tanh':
            return np.tanh(x)
        elif function == 'linear':
            return x
        else:
            return Exception('Unknown activation function')

    def fit(self, X_train, Y_train, X_test, Y_test):
        losses = []
        accuracies = []
        losses_test = []
        accuracies_test = []

        for epoch in tqdm(range(self.epochs)):
            for x, y in zip(X_train, Y_train):
                # forwarding
                y_pred = self.w @ x + self.b
                y_pred = self.activation(y_pred, 'linear')

                # back propagation
                error = y - y_pred

                # update
                self.w = self.w + x * error * self.lr
                self.b = self.b + error * self.lr
            loss_train, accuracy_train = self.evaluate(X_train, Y_train)
            loss_test, accuracy_test = self.evaluate(X_test, Y_test)

            losses.append(loss_train)
            # print(losses)
            accuracies.append(accuracy_train)
            print("acc", accuracies)
            losses_test.append(loss_test)
            accuracies_test.append(accuracy_test)
            print("acc_test", accuracies_test)

        np.save('wandb_train.npy', self.w + self.b)
        return losses, accuracies, losses_test, accuracies_test

    def predict(self, X_test):
        Y_pred = []

        for x_test in X_test:
            y_pred = x_test @ self.w + self.b
            y_pred = self.activation(y_pred)
            y_pred = np.where(y_pred > 0.5, 1, 0)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def calculate_loss(self, X_test, Y_test, metric='mae'):
        Y_pred = self.predict(X_test)
        if metric == "mae":
            return np.sum(np.abs(Y_test - Y_pred))
        elif metric == "mse":
            return np.sum((Y_test - Y_pred) ** 2)
        elif metric == "rmse":
            return np.sqrt(np.sum((Y_test - Y_pred) ** 2))

        else:
            return Exception('Unknown metric')

    def calculate_accuracy(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        accuracy = np.sum(Y_pred == Y_test) / len(Y_test)
        return accuracy

    def evaluate(self, X_test, Y_test):
        loss = self.calculate_loss(X_test, Y_test)
        accuracy = self.calculate_accuracy(X_test, Y_test)

        return loss, accuracy