import numpy as np
import matplotlib.pyplot as plt
figure, (ax1, ax2) = plt.subplots(1, 2)


class perceptron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def fit(self, X_train, y_train, epochs):
        losses = []
        learning_rate_w = 0.0001
        learning_rate_b = 0.1

        for j in range(epochs):
            for i in range(X_train.shape[0]):
                x = X_train[i]
                y = y_train[i]
                y_pred = self.predict(x)
                error = y - y_pred
                self.w = self.w + (error * x * learning_rate_w)
                self.b = self.b + (error * learning_rate_b)
                loss = self.evaluate(x, y)
                losses.append(loss)

                ax1.clear()
                ax1.scatter(X_train, y_train)
                ax1.set_xlabel('Length')
                ax1.set_ylabel('Height')

                ax1.plot(X_train, self.predict(X_train), color="red")
                plt.pause(0.01)
                ax2.clear()
                ax2.plot(losses)

    def predict(self, X_test):
        y_pred = X_test * self.w + self.b
        return y_pred

    def evaluate(self, X_test, y_test):

        y_pred = self.predict(X_test)
        error = y_test - y_pred
        loss = np.mean(np.abs(error))

        return loss
