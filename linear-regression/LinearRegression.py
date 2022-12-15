import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, num_iters=1000) -> None:
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (np.dot(X.T, (y_pred - y)))/n_samples
            db = np.sum(y_pred - y)/n_samples

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
