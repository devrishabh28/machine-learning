#  An example training of neural network using the model object.
import numpy as np
from nnfs.datasets import spiral_data
import layer
import activation_functions as af
import loss_functions as lf
import optimizers
from accuracy import AccuracyCategorical
from Model import Model

#  Create train and test dataset
X_train, y_train = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

#  Instantiate the model
model = Model()

#  Add layers
model.add(layer.LayerDense(
    2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(af.ActivationReLU())
model.add(layer.LayerDropout(0.1))
model.add(layer.LayerDense(512, 3))
model.add(af.ActivationSoftmax())

#  Set loss, optimizer and accuracy objects
model.set(
    loss=lf.CategoricalCrossEntropyLoss(),
    optimizer=optimizers.Adam(learning_rate=0.05, decay=5e-5),
    accuracy=AccuracyCategorical()
)

#  Finalize the model
model.finalize()

#  Train the model
model.train(X_train, y_train, validation_data=(
    X_test, y_test), epochs=10000, print_every=100)
