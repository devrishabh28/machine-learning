#  An example derivative for a single layer.
import numpy as np

#  Passed-in gadient from the next layer.
dvalues = np.array([
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0],
])

weights = np.array([
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]).T

inputs = np.array([
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8],
])

biases = np.array([[2, 3, 0.5]])

print(np.dot(inputs, weights) + biases)


#  Gradient with respect to inputs
dinputs = np.dot(dvalues, weights.T)
print("\n", dinputs)

dweights = np.dot(inputs.T, dvalues)
print("\n", dweights)

dbiases = np.sum(dvalues, axis=0, keepdims=True)
print("\n", dbiases)
