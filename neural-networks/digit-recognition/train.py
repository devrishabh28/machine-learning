import numpy as np
import pandas as pd
import layer
import activation_functions as af
import loss_functions as lf
import optimizers
from accuracy import AccuracyCategorical
from Model import Model

train_data = np.array(pd.read_csv('mnist_dataset/mnist_train.csv'))
m, n = train_data.shape

np.random.shuffle(train_data)
train_data = train_data.T
X_train, y_train = train_data[1:n].T, train_data[0]
X_train = (X_train - np.average(X_train)) / np.average(X_train)

test_data = np.array(pd.read_csv('mnist_dataset/mnist_test.csv'))
m, n = test_data.shape
test_data = test_data.T
X_test, y_test = test_data[1:n].T, test_data[0]
X_test = (X_test - 127.5) / 127.5


#  Instantiate the model.
model = Model()

#  Add Layers
model.add(layer.LayerDense(X_train.shape[1], 256))
model.add(af.ActivationReLU())
model.add(layer.LayerDropout(0.2))
model.add(layer.LayerDense(256, 256))
model.add(af.ActivationReLU())
model.add(layer.LayerDropout(0.6))
model.add(layer.LayerDense(256, 256))
model.add(af.ActivationReLU())
model.add(layer.LayerDropout(0.2))
model.add(layer.LayerDense(256, 10))
model.add(af.ActivationSoftmax())

#  Set loss, optimizer and accuracy objects
model.set(
    loss=lf.CategoricalCrossEntropyLoss(),
    optimizer=optimizers.Adam(decay=5e-5),
    accuracy=AccuracyCategorical()
)

#  Finalize the model
model.finalize()

#  Train the model
model.train(X_train, y_train, validation_data=(X_test, y_test),
            epochs=5, batch_size=128, print_every=100)

parameters = model.get_parameters()

model.save_parameters('mnist.parms')

model.save('mnist.model')
