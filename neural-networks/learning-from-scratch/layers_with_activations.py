#  Layers with Activation Functions.
import numpy as np
import activation_functions as af
from nnfs.datasets import spiral_data


#  Layer Dense
class LayerDense:
    def __init__(self, num_inputs, num_neurons) -> None:
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
activation1 = af.ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = af.ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
