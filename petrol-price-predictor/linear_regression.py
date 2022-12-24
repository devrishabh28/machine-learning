import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data_month.csv")


def loss_function(m, c, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].month
        y = points.iloc[i].price
        total_error += (y - (m*x + c))**2

    return total_error/float((2*len(points)))


def gradient_descent(z_current, t_current, m_current, c_current, points, L):
    m_gradient = 0
    c_gradient = 0
    t_gradient = 0
    z_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].month
        y = points.iloc[i].price

        h = (y - (z_current*(x**3) + t_current*(x**2) + m_current*x + c_current))

        z_gradient += (-(x**3) * h / n)
        t_gradient += (-(x**2) * h / n)
        m_gradient += (-x/n)*h
        c_gradient += (-1/n)*h

    z = z_current - L * z_current
    t = t_current - L * t_gradient
    m = m_current - L * m_gradient
    c = c_current - L * c_gradient

    return z, t, m, c


m = np.random.randn(1,)[0]
c = np.random.randn(1,)[0]
t = np.random.randn(1,)[0]
z = np.random.randn(1,)[0]
L = 0.01
epochs = 5001

month = data.month
mean = data.month.mean()
std = data.month.std()
data.month = (data.month - data.month.mean())/data.month.std()

for i in range(epochs):
    if (i % 1000 == 0):
        print(f"Epoch: {i}")
        print(z, t, m, c)

    z, t, m, c = gradient_descent(z, t, m, c, data, L)

print(z, t, m, c)

data.month = month


pred = [z * (((x-mean)/std)**3) + t * (((x-mean)/std)**2) +
        m*((x-mean)/std)+c for x in data.month]

err = np.mean(np.array(data.price - pred)**2)
print(err)

plt.scatter(data.month, data.price, color="black")

r = np.arange(0, 80)

plt.plot(r, [z * (((x-mean)/std)**3) + t *
         (((x-mean)/std)**2) + m*((x-mean)/std)+c for x in r])
plt.show()
