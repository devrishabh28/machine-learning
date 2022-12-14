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

    #  Regularization loss calculation
    def regularization_loss(self, layer):

        #  0 by default
        regularization_loss = 0

        #  L1 regularization - weights
        #  calculate only when factor greater than 0.
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * \
                np.sum(np.abs(layer.weights))

        #  L2 regulariztion - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                np.sum(layer.weights * layer.weights)

        #  L1 regularization - biases
        #  calculate only when factor
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * \
                np.sum(np.abs(layer.biases))

        #  L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                np.sum(layer.biases * layer.biases)

        return regularization_loss


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

        return self.dinputs


#  Binary Cross-Entropy Loss
class BinaryCrossEntropyLoss(Loss):

    #  Forward Pass
    def forward(self, y_pred, y_true):

        #  Clip data to prevent division by 0.
        #  Clip both sides to not drag mean towards any value.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        #  Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))

        sample_losses = np.mean(sample_losses, axis=-1)

        #  Return losses
        return sample_losses

    #  Bacward Pass
    def backward(self, dvalues, y_true):

        #  Number of samples
        samples = len(dvalues)

        #  Number of outputs in every sample
        outputs = len(dvalues[0])

        #  Clip data to prevent division by 0.
        #  Clip both sides to not drag mean towards any value.
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        #  Calculate gradient
        self.dinputs = -(y_true/clipped_dvalues -
                         (1 - y_true)/(1-clipped_dvalues)) / outputs

        #  Normalize gradient
        self.dinputs = self.dinputs / samples

        return self.dinputs
