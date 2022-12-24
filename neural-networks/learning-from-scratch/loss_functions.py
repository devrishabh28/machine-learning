#  Calculation of loss of a prediction
import numpy as np


#  Common Loss Class
class Loss:
    #  Calculates the data and regularizes losses
    #  of given model ouput and ground truth values.
    def calculate(self, output, y):

        #  Calculate sample losses
        sample_losses = self.forward(output, y)

        #  Calculate mean loss
        data_loss = np.mean(sample_losses)

        #  Return loss
        return data_loss


#  Categorical Cross-Entropy Loss
class CategoricalCrossEntropyLoss(Loss):

    #  Forward pass
    def forward(self, y_pred, y_true):
        #  Number of samples in a batch
        samples = len(y_pred)

        #  Clip data to prevent division by 0.
        #  Clip both sides to not drag mean towards any value.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        #  Probabilities for target values -
        #  only if categorical lables.
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        #  Mask values - pnly for one-hot encoded labels.
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        #  Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    #  Backward Pass

    def backward(self, dvalues, y_true):

        #  Number of samples.
        samples = len(dvalues)
        #  Number of lables in each sample
        labels = len(dvalues[0])

        #  If labels are sparse, tern them into one-hot vector.
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        #  Calculate gradient
        self.dinputs = -y_true / dvalues
        #  Normalize gradient
        self.dinputs = self.dinputs / samples
