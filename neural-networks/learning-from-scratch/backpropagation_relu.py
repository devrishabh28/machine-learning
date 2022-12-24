#  An example backpropagation with ReLU Activation Function.
import numpy as np

#  Passed-in values from the next layer
dvalues = np.array([
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3]
])

inputs = np.array([
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
])

weights = np.array([
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]).T

biases = np.array([[2, 3, 0.5]])

#  Forward pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

#  Backpropagation
drelu = dvalues.copy()
drelu[layer_outputs <= 0] = 0

#  Taking derivatives
dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)

print()
print(biases)
