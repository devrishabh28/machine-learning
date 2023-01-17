import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


#  Load MNIST dataset.
def load_mnist_dataset(dataset, path):

    #  Scan all the directories and create a list of labels.
    labels = os.listdir(os.path.join(path, dataset))

    #  Create lists for samples and labels.
    X = []
    y = []

    #  For each label folder
    for label in labels:
        #  For each image in given folder.
        for file in os.listdir(os.path.join(path, dataset, label)):
            #  Read the image
            image = cv2.imread(os.path.join(
                path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            #  Append image and label to the lists.
            X.append(image)
            y.append(label)

    #  Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

#  MNIST dataset (train + test)


def create_data_mnist(path):

    #  Load both sets separately
    X_train, y_train = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    #  Return all the data
    return X_train, y_train, X_test, y_test
