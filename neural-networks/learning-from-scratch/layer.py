#  A class for a single layer of neurons
import numpy as np


#  Dense Layer
class LayerDense:
    #  Layer Initialization
    def __init__(self, num_inputs, num_neurons) -> None:
        #  Initialize weights and biases
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    #  Forward pass
    def forward(self, inputs):
        #  Remember the input values
        self.inputs = inputs
        #  Calculate the output values from inputs, weights and biases.
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    #  Backward Pass
    def backward(self, dvalues):
        #  Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbaises = np.sum(dvalues, axis=0, keepdims=True)

        #  Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
