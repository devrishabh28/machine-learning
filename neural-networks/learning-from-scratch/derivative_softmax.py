#  Derivative of the Softmax Activation function.
import numpy as np

softmax_output = [0.7, 0.1, 0.2]

d_values = [1, 1, 1]

softmax_output = np.array(softmax_output).reshape(-1, 1)

print(softmax_output)

print(softmax_output * np.eye(softmax_output.shape[0]))

print(np.diagflat(softmax_output))

print(np.dot(softmax_output, softmax_output.T))

print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))

print()

print(np.dot(np.diagflat(softmax_output) -
      np.dot(softmax_output, softmax_output.T), np.array(d_values)))
