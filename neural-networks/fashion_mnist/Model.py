#  Model Class for Neural Network
import numpy as np
import layer
import activation_functions as af
import loss_functions as lf
from math import ceil
import pickle
import copy


class Model:

    def __init__(self) -> None:
        #  Create a list of network objects.
        self.layers = []

        # Softmax classifier's output object
        self.softmax_classifier_output = None

    #  Add objects to the model.
    def add(self, layer):
        self.layers.append(layer)

    #  Set loss and optimizer
    def set(self, *, loss=None, optimizer=None, accuracy=None):

        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
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
        if self.loss is not None:
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
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

        #  Initialze accuracy object
        self.accuracy.init(y)

        #  Default value if batch size if not set.
        train_steps = 1

        #  Calculate number of steps
        if batch_size is not None:
            train_steps = ceil(len(X) / batch_size)

        #  Training in loop.
        for epoch in range(1, epochs+1):

            #  Print epoch number
            print(f'epoch:{epoch}')

            #  Reset accumulated values in loss and accuracy objects.
            self.loss.new_pass()
            self.accuracy.new_pass()

            #  Iterate over steps.
            for step in range(train_steps):

                #  If batch size is not set
                #  train using one step and full dataset.
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                #  Otherwise slice a batch.
                else:
                    batch_X = X[step * batch_size: (step+1) * batch_size]
                    batch_y = y[step * batch_size: (step+1) * batch_size]

                #  Perform forward pass
                output = self.forward(batch_X, training=True)

                #  Calculate loss
                data_loss, regularization_loss = self.loss.calculate(
                    output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                #  Get predictions and calculate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                #  Perform backward pass
                self.backward(output, batch_y)

                #  Optimize (update parameters)
                self.optimizer.updateParameters(self.trainable_layers)

                #  Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

                #  Get and print epoch loss and accuracy
                epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(
                    include_regularization=True)
                epoch_loss = epoch_data_loss + epoch_regularization_loss
                epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

        #  If there is validation data present.
        if validation_data is not None:

            #  Evaluate the model
            self.evaluate(*validation_data, batch_size=batch_size)

    #  Evaluates the model using passed-in dataset.
    def evaluate(self, X_val, y_val, *, batch_size=None):

        #  Default value if batch size is not set.
        validation_steps = 1

        #  Calculate number of steps
        if batch_size is not None:
            validation_steps = ceil(len(X_val) / batch_size)

        #  Reset accumulated values in loss
        #  and accuracy objects.
        self.loss.new_pass()
        self.accuracy.new_pass()

        #  Iterate over steps
        for step in range(validation_steps):

            #  If batch size is not set -
            #  train using one step and full dataset.
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            #  Otherwise slice a batch
            else:
                batch_X = X_val[step * batch_size: (step+1) * batch_size]
                batch_y = y_val[step * batch_size: (step+1) * batch_size]

            #  Perform forward pass
            output = self.forward(batch_X, training=False)

            #  Calculate loss
            loss = self.loss.calculate(output, batch_y)

            #  Get predictions and calculate accuracy.
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)

        #  Get and print validation loss and accuracy.
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    #  Retrieves and returns parameters of trainable layers.
    def get_parameters(self):

        #  Create a list for paramters.
        parameters = []

        #  Iterate trainable layers and get their paramters.
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        #  Return a list
        return parameters

    #  Updates the model with new paramters.
    def set_parameters(self, parameters):

        #  Iterate over the paramters and layers
        #  and update each layer witheach set of parameters.
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    #  Saves the parameter to a file.
    def save_parameters(self, path):

        #  Open a file in the binar-write mode
        #  and save parameters to it.
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    #  Loads the weights and updates a model instance with them.
    def load_paramters(self, path):

        #  Open file in the binary-read mode,
        #  load weights and update trainable layers.
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    #  Saves the model.
    def save(self, path):

        #  Make a deep copy of current model instance.
        model = copy.deepcopy(self)

        #  Reset accumulated values in loss and accuracy objects.
        model.loss.new_pass()
        model.accuracy.new_pass()

        #  Remove data from the input layer
        #  and gradients from the loss object.
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        #  For eah layer remove inputs, output and dinput properties.
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        #  Open a file in the write-binary mode
        #  and save the model.
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    #  Loads and returns a model.
    @staticmethod
    def load(path):

        #  open file in the binar-read mode,
        #  and load the mode.
        with open(path, 'rb') as f:
            model = pickle.load(f)

        #  Return a model.
        return model

    #  Predicts on the samples
    def predict(self, X, *, batch_size=None):

        #  Default value if batch size is not set.
        prediction_steps = 1

        #  Calculate the number of steps
        if batch_size is not None:
            prediction_steps = ceil(prediction_steps/batch_size)

        #  Model outputs
        output = []

        #  Iterate over steps
        for step in range(prediction_steps):

            #  If batch size is not set -
            #  train using one step and full dataset.
            if batch_size is None:
                batch_X = X

            #  Otherwise slice a batch.
            else:
                batch_X = X[step*batch_size:(step + 1)*batch_size]

            #  Perform the forward pass
            batch_output = self.forward(batch_X)

            #  Append batch prediction to the list of predictions.
            output.append(batch_output)

        #  Stack and return results
        return np.vstack(output)
