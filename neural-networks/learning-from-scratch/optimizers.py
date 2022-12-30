#  Library for optimizers.
import numpy as np


#  Stochastic Gradient Descent Optimizer (SGD)
class StochasticGradientDescent:

    #  Initialize optimizer
    def __init__(self, learning_rate=1, decay=0., momentum=0.) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    #  Update parameters
    def updateParameters(self, layers):

        #  Pre-Update Parameters
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1 + self.decay * self.iterations)

        #  Update Parameters
        for layer in layers:

            #  If momentum is used
            if self.momentum:

                #  If layer does not contain momentum arrays,
                #  create them filled with zero.
                if not hasattr(layer, 'weight_momentums'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)

                #  Build weight updates by momentum
                weight_updates = self.momentum * layer.weight_momentums - \
                    self.current_learning_rate * layer.dweights
                layer.weight_momentums = weight_updates

                #  Build bias updates by momentum
                bias_updates = self.momentum * layer.bias_momentums - \
                    self.current_learning_rate * layer.dbiases
                layer.bias_momentums = bias_updates

            #  Vanilla SGD updates
            else:
                weight_updates = -self.current_learning_rate * layer.dweights
                bias_updates = -self.current_learning_rate * layer.dbiases

            #  Update weights and biases using either
            #  vanilla or momentum updates.
            layer.weights += weight_updates
            layer.biases += bias_updates

        #  Post-Update Parameters
        self.iterations += 1


#  Adaptive Gradient Optimizer (AdaGrad)
class AdaptiveGradient:

    #  Initialize optimizer
    def __init__(self, learning_rate=1, decay=0., epsilon=1e-7) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    #  Update parameters
    def updateParameters(self, layers):

        #  Pre-Update Parameters
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1 + self.decay * self.iterations)

        #  Update Parameters
        for layer in layers:
            if not hasattr(layer, 'weight_cache'):
                #  If layer doesn't contain cache arrays,
                #  create them filled with zeroes.
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            #  Update cache with squared current gradients
            layer.weight_cache += layer.dweights**2
            layer.bias_cache += layer.dbiases**2

            #  Vanilla SGD parameter update with
            #  normalization using square rooted cache
            layer.weights += -self.current_learning_rate * \
                layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * \
                layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

        #  Post-Update Parameters
        self.iterations += 1


#  Root Mean Square Propagation Optimizer (RMSProp)
class RootMeanSquarePropagation:

    #  Initialze optimizer
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    #  Update parametrs
    def updateParameters(self, layers):

        #  Pre-Update Parameters
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1 + self.decay * self.iterations)

        #  Update Parameters
        for layer in layers:
            if not hasattr(layer, 'weight_cache'):
                #  If layer doesn't contain cache arrays,
                #  create them filled with zeroes.
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            #  Update cache with squared current gradients.
            layer.weight_cache = self.rho * layer.weight_cache + \
                (1 - self.rho) * layer.dweights**2
            layer.bias_cache = self.rho * layer.bias_cache + \
                (1 - self.rho) * layer.dbiases**2

            #  Vanilla SGD parameter update with
            #  normalization using square rooted cache
            layer.weights += -self.current_learning_rate * \
                layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * \
                layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

        #  Post-Update Parameters
        self.iterations += 1
