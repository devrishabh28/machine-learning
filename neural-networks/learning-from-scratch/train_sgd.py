#  Training nerual network using Stochastic Gradient Descent.
import numpy as np
from nnfs.datasets import spiral_data
import layer
import activation_functions as af
import optimizers

#  Create dataset
X, y = spiral_data(samples=100, classes=3)

#  Create a dense layer with 2 input features and 6 output features.
dense1 = layer.LayerDense(2, 64)

#  Create ReLU activation
activation1 = af.ActivationReLU()

#  Create second dense layer with 64 input features and 3 output values.
dense2 = layer.LayerDense(64, 3)

#  Create Softmax Classifier's combiner loss and activation.
loss_activation = af.SoftmaxClassifier()

#  Create optimizer
optimizer = optimizers.StochasticGradientDescent(decay=1e-3, momentum=0.9)

#  Train in loop
for epoch in range(10001):

    #  Perform a forward pass of the training data through this layer.
    dense1.forward(X)

    #  Perform a forward pass through activation function
    #  takes the output of first dense layer as input.
    activation1.forward(dense1.output)

    #  Perform a forward pass through the second dense layer
    #  takes the output of the activation function as input.
    dense2.forward(activation1.output)

    #  Perform a forward pass through the activation/loss function
    #  takes the output of the second dense layer as input and returns loss.
    loss = loss_activation.forward(dense2.output, y)

    #  Calculate accuracy from the output of activation2 and targets
    #  calculate values along first axis.
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(
            f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.2f}, lr: {optimizer.current_learning_rate}'
        )

    #  Back Propagation
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #  Update weights and biases
    optimizer.updateParameters([dense1, dense2])
