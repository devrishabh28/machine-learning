import numpy as np
import layer
import activation_functions as af
import loss_functions as lf
import optimizers
from accuracy import AccuracyCategorical
from Model import Model
from load_data import create_data_mnist

#  Create dataset
X_train, y_train, X_test, y_test = create_data_mnist('fashion_mnist_images')

#  Shuffle the training dataset
keys = np.array(range(X_train.shape[0]))
np.random.shuffle(keys)
X_train = X_train[keys]
y_train = y_train[keys]

#  Scale features
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

#  Reshape to vectors.
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

#  Instantiate the model
model = Model()

#  Add layers
model.add(layer.LayerDense(X_train.shape[1], 128))
model.add(af.ActivationReLU())
model.add(layer.LayerDense(128, 128))
model.add(af.ActivationReLU())
model.add(layer.LayerDense(128, 10))
model.add(af.ActivationSoftmax())

#  Set loss, optimizer and accuracy objects
model.set(
    loss=lf.CategoricalCrossEntropyLoss(),
    accuracy=AccuracyCategorical()
)

#  Finalize the model
model.finalize()

#  Set model with parameters.
model.load_paramters('fashion_mnist.parms')

#  Evaluate the model
model.evaluate(X_test, y_test, batch_size=128)
