#  Model Class for Neural Network
import numpy as np
import layer
import activation_functions as af
import loss_functions as lf


class Model:

    def __init__(self, layers=[]) -> None:
        #  Create a list of network objects.
        self.layers = layers

        # Softmax classifier's output object
        self.softmax_classifier_output = None

    #  Add objects to the model.
    def add(self, layer):
        self.layers.append(layer)

    #  Set loss and optimizer
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    #  Finalize the model.
    def finalize(self):
        #  Create and set the input layer
        self.input_layer = layer.LayerInput()

        #  Count all the objects.
        layer_count = len(self.layers)

        #  Initialize a list containing trainable layers.
        self.trainable_layers = []

        #  Iterate the objects
        for i in range(layer_count):

            #  If it's the first layer,
            #  the previous layer object is the input layer.
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            #  All layers except for the first and the last.
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            #  The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        #  Update loss opject with trainable layers.
        self.loss.remember_trainable_layers(self.trainable_layers)

        #  If ouput activation is Softmax and
        #  loss function is Categorical Cross-Entropy
        #  create and object of combined activation
        # and loss function containing faster gradient calculation.
        if isinstance(self.layers[-1], af.ActivationSoftmax) and isinstance(self.loss, lf.CategoricalCrossEntropyLoss):
            #  Create an object of combined activation
            #  and loss functions.
            self.softmax_classifier_output = af.SoftmaxClassifier()

    #  Forward Pass
    def forward(self, X, training=False):

        #  Setting the input layer.
        self.input_layer.forward(X, training)

        #  Call forward method of every object in a chain
        #  Pass output of the previous object as a parameter.
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        #  Return the output of the last layer.
        return layer.output

    #  Backward Pass
    def backward(self, output, y):

        #  If Softmax Classifier
        if self.softmax_classifier_output is not None:
            #  First call backward method
            #  on the combined activation/loss
            #  this will set dinputs property.
            self.softmax_classifier_output.backward(output, y)

            #  Since backward method of the last layer will not be called
            #  which is Softmax Activtion
            #  as combined activation/loss object is used,
            #  set dinputs in this object.
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            #  Call backward method going through
            #  all the objects but the last one
            #  in reversed order passing dinputs as a parameter.
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        #  First call backward method on the less
        #  this will set dinputs property that the
        #  last layer will try to access shortly.
        self.loss.backward(output, y)

        #  Call backward method going through all the objects
        #  in reversed order passing dinputs as a parameter.
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    #  Train the model.
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        #  Initialze accuracy object
        self.accuracy.init(y)

        #  Training in loop.
        for epoch in range(1, epochs+1):

            #  Perform forward pass
            output = self.forward(X, training=True)

            #  Calculate loss
            data_loss, regularization_loss = self.loss.calculate(
                output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            #  Get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            #  Perform backward pass
            self.backward(output, y)

            #  Optimize (update parameters)
            self.optimizer.updateParameters(self.trainable_layers)

            #  Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')

        #  If there is validation data present.
        if validation_data is not None:

            X_val, y_val = validation_data

            #  Perform forward pass
            output = self.forward(X_val, training=False)

            #  Calculate loss
            loss = self.loss.calculate(output, y_val)

            #  Get predictions and calculate accuracy.
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Print a summary
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')
