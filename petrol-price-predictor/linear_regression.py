import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_month.csv")


def loss_function(m, c, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].month
        y = points.iloc[i].price
        total_error += (y - (m*x + c))**2

    return total_error/float((2*len(points)))


def gradient_descent(m_current, c_current, points, L):
    m_gradient = 0
    c_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].month
        y = points.iloc[i].price

        m_gradient += (-x/n)*(y - (m_current*x + c_current))
        c_gradient += (-1/n)*(y - (m_current*x + c_current))

    m = m_current - L * m_gradient
    c = c_current - L * c_gradient

    return m, c


m = 0
c = 0
L = 0.001
epochs = 10000

for i in range(epochs):
    if (i % 1000 == 0):
        print(f"Epoch: {i}")
    m, c = gradient_descent(m, c, data, L)

print(m, c)

plt.scatter(data.month, data.price, color="black")
plt.plot(list(range(0, 80)), [m*x+c for x in range(0, 80)])
plt.show()
