import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)


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
                x = x.reshape(-1, 2)
                y = y_train[i]
                y_pred = self.predict(x)
                error = y - y_pred
                self.w = self.w + error * x * learning_rate_w
                self.b = self.b + error * learning_rate_b
                loss = self.evaluate(x, y)
                losses.append(loss)
                X1 = x[:, 0]
                X2 = x[:, 1]
                xx, yy = np.meshgrid(X1, X2)
                # plot the results
                ax1.clear()
                ax1.scatter(X1, X2, y)
                surface = xx * self.w[0, 0] + yy * self.w[0, 1]
                ax1.plot_surface(xx, yy, surface, cmap="viridis")
                ax1.set_xlabel('$business$', fontsize=10)
                ax1.set_ylabel('$highways$', fontsize=10)
                ax1.set_zlabel('$Tax$', fontsize=10)
                plt.pause(0.01)
                ax2.clear()
                ax2.plot(losses)

        plt.show()

    def predict(self, X_test):

        y_pred = X_test * self.w + self.b
        return y_pred

    def evaluate(self, X_test, y_test):

        y_pred = self.predict(X_test)
        error = y_test - y_pred
        loss = np.mean(np.abs(error))

        return loss
