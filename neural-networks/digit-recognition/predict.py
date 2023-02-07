import numpy as np
from Model import Model
import matplotlib.pyplot as plt
import cv2

#  Read an image
image_data = cv2.imread('2.jpeg', cv2.IMREAD_GRAYSCALE)

#  Resize to the smae size as Fashion MNIST images.
image_data = cv2.resize(image_data, (28, 28))

# Invert image colors
image_data = 255 - image_data

#  Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) -
              127.5) / 127.5

#  Load the model
model = Model.load('mnist.model')

#  Predict on the image
confidences = model.predict(image_data)

#  Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

#  Get label name from label index
prediction = predictions[0]

print(f'The following image is {prediction}')

#  Plotting the image
plt.imshow(cv2.cvtColor(cv2.imread(
    '2.jpeg', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))
plt.show()
