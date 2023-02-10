import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
X = X[:, np.newaxis, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

plt.scatter(X_train, y_train)
plt.plot(np.linspace(-0.15, 0.15, 1000).reshape(-1, 1),
         model.predict(np.linspace(-0.15, 0.15, 1000).reshape(-1, 1)), 'r')
plt.show()
