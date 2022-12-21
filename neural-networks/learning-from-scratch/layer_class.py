#  Making a class for layers
import numpy as np
from nnfs.datasets import spiral_data


#  Dense Layer
class LayerDense:
    #  Layer Initialization
    def __init__(self, num_inputs, num_neurons) -> None:
        #  Initialize weights and biases
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    #  Forward pass
    def forward(self, inputs):
        #  Calculate the output values from inputs, weights and biases.
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


#  Create dataset
X, y = spiral_data(samples=100, classes=3)

#  Create Dense Layer with 2 input feautures and 3 output values.
dense1 = LayerDense(2, 3)

#  Perform a forward pass of the training data trhough this layer.
dense1.forward(X)

print(dense1.output[:5])
