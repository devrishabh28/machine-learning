#  Binary Logistic Regression using neural network.
import numpy as np
from nnfs.datasets import spiral_data
import layer
import activation_functions as af
import loss_functions as lf
import optimizers

#  Create dataset
X_train, y_train = spiral_data(samples=100, classes=2)

#  Reshape labels to be a list of lists
#  Inner list contains one output (either 0 or 1)
#  per each output neuron, 1 in this case.
y_train = y_train.reshape(-1, 1)

#  Create dense layer with 2 input features and 64 output values.
dense1 = layer.LayerDense(
    2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

#  Create ReLU activation
activation1 = af.ActivationReLU()

#  Create a second dense layer with 64 input features and 1 output value.
dense2 = layer.LayerDense(64, 1)

#  Create Sigmoid activation
activation2 = af.ActivationSigmoid()

#  Create loss function
loss_function = lf.BinaryCrossEntropyLoss()

#  Create optimizer
optimizer = optimizers.Adam(decay=5e-7)

#  Train in loop
for epoch in range(10001):
    #  Perform a forward pass of the training data through this layer.
    dense1.forward(X_train)

    #  Perform a forward pass through activation function
    #  takes the output of the first dense layer as inputs.
    activation1.forward(dense1.output)

    #  Perform a forward pass through the second dense layer
    #  takes the outputs of the dropout layer as inputs.
    dense2.forward(activation1.output)

    # Perform a forward pass through activation function
    # takes the output of second dense layer here
    activation2.forward(dense2.output)

    #  Perform a forward pass through activation/loss function
    #  takes the output of the second dense layer as inputs.
    data_loss = loss_function.calculate(activation2.output, y_train)

    #  Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(
        dense1) + loss_function.regularization_loss(dense2)

    #  Calcuate overall loss
    loss = data_loss + regularization_loss

    #  Calculate accuracy from the output of activation2 and targets
    #  calculate values along first axis
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y_train)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    #  Bacward pass
    loss_function.backward(activation2.output, y_train)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #  Update weights and biases
    optimizer.updateParameters([dense1, dense2])


#  Validate the model

#  Create test dataset
X_test, y_test = spiral_data(samples=100, classes=2)

#  Reshape labels to be a list of lists
#  Inner list contains one output (either 0 or 1)
#  per each output neuron, 1 in this case.
y_test = y_test.reshape(-1, 1)

#  Perform a forward pass of the testing data through this layer.
dense1.forward(X_test)

#  Perform a forward pass through activation function
#  takes the output of the first dense layer as inputs.
activation1.forward(dense1.output)

#  Perform a forward pass through the second dense layer
#  takes the outputs of the dropout layer as inputs.
dense2.forward(activation1.output)

# Perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward(dense2.output)

#  Perform a forward pass through activation/loss function
#  takes the output of the second dense layer as inputs.
data_loss = loss_function.calculate(activation2.output, y_test)

#  Calculate regularization penalty
regularization_loss = loss_function.regularization_loss(
    dense1) + loss_function.regularization_loss(dense2)

#  Calcuate overall loss
loss = data_loss + regularization_loss

#  Calculate accuracy from the output of activation2 and targets
#  calculate values along first axis
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
