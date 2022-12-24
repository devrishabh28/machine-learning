#  An example backpropagation.
import numpy as np
from nnfs.datasets import spiral_data
import layer
import activation_functions as af

X, y = spiral_data(samples=100, classes=3)

dense1 = layer.LayerDense(2, 3)
activation1 = af.ActivationReLU()

dense2 = layer.LayerDense(3, 3)
loss_activation = af.SoftmaxClassifier()

#  Forward Propagation
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

print('loss:', loss)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

print('acc:', accuracy)

#  Back Propagation
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

#  Print gradients
print("DENSE 1")
print(dense1.dweights)
print(dense1.dbaises)
print("\nDENSE 2")
print(dense2.dweights)
print(dense2.dbaises)
