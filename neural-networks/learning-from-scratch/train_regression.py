#  Regression using neural network.
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import sine_data
import layer
import activation_functions as af
import loss_functions as lf
import optimizers


#  Create dataset
X_train, y_train = sine_data()

#  Create Dense Layer with 1 input feature and 64 output values
dense1 = layer.LayerDense(1, 64)

#  Create ReLU activation
activation1 = af.ActivationReLU()

#  Create a second Dense Layer with 64 input features and 64 output value.
dense2 = layer.LayerDense(64, 64)

#  Create ReLU activation
activation2 = af.ActivationReLU()

#  Create a third Dense Layer with 64 input features and 1 output value.
dense3 = layer.LayerDense(64, 1)

#  Create Linear activation
activation3 = af.ActivationLinear()

#  Create loss function
loss_function = lf.MeanSquaredErrorLoss()

#  Create optimizer
optimizer = optimizers.Adam(learning_rate=0.004, decay=1e-4)

#  Accuracy precision for accuracy calculation
accuracy_precision = np.std(y_train) / 250

#  Train in loop
for epoch in range(10001):

    #  Perform a forward pass of the training data through thus layer.
    dense1.forward(X_train)

    #  Perform a forward pass through activation function
    #  takes the output of the first dense layer as input.
    activation1.forward(dense1.output)

    #  Perform a forward pass through second Dense Layer
    #  takes the output of the activation function of first layer as input.
    dense2.forward(activation1.output)

    #  Perform a forward pass through activation function
    #  takes the output of the second dense layer here.
    activation2.forward(dense2.output)

    #  Perform a forward pass through third Dense Layer
    #  takes the output of the activation function of second layer as input.
    dense3.forward(activation2.output)

    #  Perform a forward pass through activation function
    #  takes the output of the third dense layer here.
    activation3.forward(dense3.output)

    #  Calculate the data loss
    data_loss = loss_function.calculate(activation3.output, y_train)

    #  Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(
        dense1) + loss_function.regularization_loss(dense2)

    #  Calculate overall loss
    loss = data_loss + regularization_loss

    #  Calculate accuracy from output of activation2 and taragets.
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y_train) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    #  Backward Pass
    loss_function.backward(activation3.output, y_train)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #  Update weights and biases.
    optimizer.updateParameters([dense1, dense2])

#  Ceating test dataset
X_test, y_test = sine_data()

#  Forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

#  Plotting the output.
plt.plot(X_test, y_test, linewidth=2)
plt.plot(X_test, activation3.output)
plt.show()
