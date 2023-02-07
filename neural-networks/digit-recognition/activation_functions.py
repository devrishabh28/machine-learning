#  Some popular activation functions.
import numpy as np
from loss_functions import CategoricalCrossEntropyLoss


#  Step Activation
class ActivationStep:
    #  Forward Pass
    def forward(self, inputs, training=False):
        self.output = np.heaviside(inputs, 1)
        return self.output


#  Linear Activation
class ActivationLinear:
    #  Forward Pass
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = inputs
        return self.output

    #  Backward Pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    #  Calcuate predictions for output
    def predictions(slef, outputs):
        return outputs


#  Sigmoid Activation
class ActivationSigmoid:
    #  Forward Pass
    def forward(self, inputs, training=False):
        self.output = 1/(1 + np.exp(-inputs))
        return self.output

    #  Backward Pass
    def backward(self, dvalues):
        #  Derivative - calculates from output of the sigmoid function.
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1


#  ReLU Activation
class ActivationReLU:
    #  Forward Pass
    def forward(self, inputs, training=False):
        self.output = np.maximum(0, inputs)
        return self.output

    #  Backward Pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0

    #  Calcuate predictions for output
    def predictions(slef, outputs):
        return outputs


#  Softmax Activation
class ActivationSoftmax:
    #  Forward Pass
    def forward(self, inputs, training=False):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    #  Backward Pass
    def backward(self, dvalues):

        #  Create uninitailized array
        self.dinputs = np.empty_like(dvalues)

        #  Enumerate oupts and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #  Flatten output array
            single_output = single_output.reshape(-1, 1)
            #  Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(
                single_dvalues) - np.dot(single_dvalues, single_dvalues.T)

            #  Calculate sample-wise gradient
            #  and add it to the array of sample gradients.
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


#  Softmax Classifier - combined Softmax ACtivation
#  and Categorical Cross-Entropy Loss for faster backward step.
class SoftmaxClassifier:

    #  Creates activation and loss function objects
    def __init__(self) -> None:
        self.activation = ActivationSoftmax()
        self.loss = CategoricalCrossEntropyLoss()

    #  Forward Pass
    def forward(self, inputs, y_true, include_regularization=False, training=False):
        #  Output layer's activation function.
        self.activation.forward(inputs)

        #  Set the output
        self.output = self.activation.output

        #  Calculate and return loss value.
        return self.loss.calculate(self.output, y_true, include_regularization=include_regularization)

    #  Backward Pass
    def backward(self, dvalues, y_true):

        #  Number of samples
        samples = len(dvalues)

        #  If labels are one-hot encoded,
        #  turn them into discrete values.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        #  Copy in order to safely modify dvalues.
        self.dinputs = dvalues.copy()

        #  Calculate gradient
        self.dinputs[range(samples), y_true] -= 1

        #  Normalize gradient
        self.dinputs = self.dinputs / samples
