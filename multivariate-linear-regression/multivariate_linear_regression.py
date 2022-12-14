import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def normalize_features(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu)/sigma
    X[0] = 1
    return X, mu, sigma


def gradient_descent(X, z, theta, L):
    m = len(z)
    gradient = np.dot(np.dot(theta.T, X.T) - np.array(z.T), X)/m
    theta = theta - L*np.array(gradient).T
    return theta


data = pd.read_csv('data.csv')
x = data['x']
y = data['y']
z = data['z']
ax = plt.axes(projection='3d')

X = pd.DataFrame([np.ones(((len(x)))), x, y]).T


X, mu, sigma = normalize_features(X)

alpha = 0.6
epochs = 1000
theta = np.random.randn(3, 1)

for i in range(epochs):
    if (i % 100 == 0):
        print(f"Epoch: {i}")
    t = gradient_descent(X, z, theta, alpha)
    theta = t

print(pd.Series(theta.flatten(), name="Theta"))
print()


def predict(x, y):
    return np.dot(theta.T, [[1], [(x-mu[1])/sigma[1]], [(y-mu[2])/sigma[2]]]).sum()


x1 = int(input("Enter the area of the house: "))
y1 = int(input("Enter the number of bedrooms: "))
print("The predicted price of the house is {:.2f}".format(predict(x1, y1)))


ax.scatter3D(x, y, z)
ax.set_xlabel('Area')
ax.set_ylabel('Bedrooms')


for rooms in range(1, 8):
    x = np.arange(800, 5501, 10)
    y = [rooms]*len(x)
    z = []

    for i in range(len(x)):
        z.append(predict(x[i], y[i]))

    ax.plot(x, y, z)

plt.show()
