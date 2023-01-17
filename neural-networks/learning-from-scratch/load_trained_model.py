import numpy as np
from Model import Model
from nnfs.datasets import spiral_data


#  Create train and test dataset
X_train, y_train = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

#  Load the model
model = Model.load('spiral_data.model')

#  Evaluate the model
model.evaluate(X_test, y_test, batch_size=128)
