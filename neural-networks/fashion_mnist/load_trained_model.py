import numpy as np
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

#  Load the model
model = Model.load('fashion_mnist.model')

#  Evaluate the model
model.evaluate(X_test, y_test, batch_size=128)
