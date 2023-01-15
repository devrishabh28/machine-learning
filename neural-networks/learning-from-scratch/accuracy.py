#  A class for calculating accuracies
import numpy as np

#  Common accuracy class
class Accuracy:

    #  Calculates accuracy
    #  given predictions and ground truth values.
    def calculate(self, predictions, y):

        #  Get comparison results.
        comparisions = self.compare(predictions, y)

        #  Calculate the accuracy.
        accuracy = np.mean(comparisions)

        #  Return accuracy
        return accuracy


#  Accuracy calculation for regression model.
class AccuracyRegression(Accuracy):

    def __init__(self) -> None:
        super().__init__()

        #  Create precision property
        self.precision = None

    #  Calculates precision value
    #  based on passed-in ground truth.
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    #  Compares predictions to the ground truth values.
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
    

#  Accuracy calculation for classification model.
class AccuracyCategorical(Accuracy):

    def __init__(self,*, binary=False) -> None:
        super().__init__()
        #  Binary mode
        self.binary = binary

    #  No initialization is needed
    def init(self, y):
        pass

    #  Compares predictions to the ground truth values.
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y