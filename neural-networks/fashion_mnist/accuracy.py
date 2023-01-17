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

        if hasattr(self, 'accumulated_sum'):
            #  Add accumulated sum of matching values and sample count.
            self.accumulated_sum += np.sum(comparisions)
            self.accumulated_count += len(comparisions)

        #  Return accuracy
        return accuracy

    #  Calculated accumulated accuracy.
    def calculate_accumulated(self):

        #  Calculate accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        #  Return accuracy
        return accuracy

    #  Reset variables for accumulated accuracy.
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


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

    def __init__(self, *, binary=False) -> None:
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
