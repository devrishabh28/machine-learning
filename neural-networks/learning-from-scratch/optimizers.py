#  Library for optimizers.
import numpy as np


#  Stochastic Gradient Descent Optimizer
class StochasticGradientDescent:

    #  Initialize optimizer
    def __init__(self, learning_rate=1, decay=0.) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    #  Update parameters
    def updateParameters(self, layers):

        #  Pre-Update Parameters
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1 + self.decay * self.iterations)

        #  Update Parameters
        for layer in layers:
            layer.weights += -self.current_learning_rate * layer.dweights
            layer.biases += -self.current_learning_rate * layer.dbiases

        #  Post-Update Parameters
        self.iterations += 1
