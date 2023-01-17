#  A class for a single layer of neurons
import numpy as np


#  Dense Layer
class LayerDense:
    #  Layer Initialization
    def __init__(self, num_inputs, num_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0) -> None:
        #  Initialize weights and biases
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        #  Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    #  Forward pass
    def forward(self, inputs, training=False):
        #  Remember the input values
        self.inputs = inputs
        #  Calculate the output values from inputs, weights and biases.
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    #  Backward Pass
    def backward(self, dvalues):
        #  Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #  Gradients on regularization
        #  L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        #  L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        #  L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        #  L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        #  Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

        return self.dinputs

    #  Retrieve layer parameters.
    def get_parameters(self):
        return self.weights, self.biases

    #  Set weights and biases in a layer instance.
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


#  Layer Dropout
class LayerDropout:

    #  Layer initialization
    def __init__(self, rate) -> None:
        #  Store rate
        self.rate = 1 - rate

    #  Forward Pass
    def forward(self, inputs, training=False):
        #  Save input values
        self.inputs = inputs

        #  If not in the training mode - return values.
        if not training:
            self.output = inputs.copy()
            return

        #  Generate and save scaled mask
        self.binary_mask = np.random.binomial(
            1, self.rate, inputs.shape) / self.rate

        #  Apply mask to output values
        self.output = inputs * self.binary_mask

        return self.output

    #  Backward Pass
    def backward(self, dvalues):
        #  Gradient on values
        self.dinputs = dvalues * self.binary_mask


#  Layer Input
class LayerInput:

    #  Forward pass
    def forward(self, inputs, training=False):
        self.output = inputs