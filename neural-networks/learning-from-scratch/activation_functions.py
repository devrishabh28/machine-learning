#  Some popular activation functions.
import numpy as np


#  Step Activation
class ActivationStep:
    #  Forward Pass
    def forward(self, inputs):
        self.output = np.heaviside(inputs, 1)
        return self.output


#  Linear Activation
class ActivationLinear:
    #  Forward Pass
    def forward(self, inputs):
        self.output = inputs
        return self.output


#  Sigmoid Activation
class ActivationSigmoid:
    #  Forward Pass
    def forward(self, inputs):
        self.output = 1/(1 + np.exp(-inputs))
        return self.output


#  ReLU Activation
class ActivationReLU:
    #  Forward Pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output


#  Softmax Activation
class ActivationSoftmax:
    #  Forward Pass
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output