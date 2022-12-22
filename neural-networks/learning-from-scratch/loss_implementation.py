#  Implementing loss in the neural network.
import numpy as np
import layer
import activation_functions as af
import loss_functions
from nnfs.datasets import spiral_data

#  Create dataset
X, y = spiral_data(samples=100, classes=3)

dense1 = layer.LayerDense(2, 3)
activation1 = af.ActivationReLU()

dense2 = layer.LayerDense(3, 3)
activation2 = af.ActivationSoftmax()

loss_function = loss_functions.CategoricalCrossEntropyLoss()


dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)

print("Loss: ", loss)
