import numpy as np
from Model import Model
from load_data import create_data_mnist

#  Create dataset
X_train, y_train, X_test, y_test = create_data_mnist('fashion_mnist_images')

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

#  Shuffle the training dataset
keys = np.array(range(X_test.shape[0]))
np.random.shuffle(keys)
X_test = X_test[keys]
y_test = y_test[keys]

#  Scale features
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

#  Reshape to vectors.
X_test = X_test.reshape(X_test.shape[0], -1)

#  Load the model
model = Model.load('fashion_mnist.model')

#  Evaluate the model
model.evaluate(X_test, y_test, batch_size=128)

confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)

for prediction in predictions:
    print(fashion_mnist_labels[prediction])
