#  Library for optimizers.
import numpy as np


#  Stochastic Gradient Descent Optimizer
class StochasticGradientDescent:

    #  Initialize optimizer
    def __init__(self, learning_rate=0.1) -> None:
        self.learning_rate = learning_rate

    #  Update parameters
    def updateParameters(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
