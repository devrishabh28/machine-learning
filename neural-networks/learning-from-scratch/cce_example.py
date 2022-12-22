#  Categorical Cross Entropy as a loss function for softmax output.
import numpy as np

#  An example output from the output layer of the neural network.
softmax_output = [0.7, 0.1, 0.2]

#  Ground truth
target_ouput = [1, 0, 0]

#  Catagorical Cross Entropy
loss = -(
    np.log(softmax_output[0])*target_ouput[0] +
    np.log(softmax_output[1])*target_ouput[1] +
    (np.log(softmax_output[2])*target_ouput[2])
)

print(loss)

#  Class Tagrget Examples
softmax_output = [
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
]

class_targets = [0, 1, 1]

print("\nClass targets")
for targ_index, distribution in zip(class_targets, softmax_output):
    print(distribution[targ_index])

softmax_output = np.array(softmax_output)
print("\nClass Targets")
print(softmax_output[range(len(softmax_output)), class_targets])

loss = -np.log(softmax_output[range(len(softmax_output)), class_targets])

print(f"Loss = {loss}")

print(f"Average Loss = {np.mean(loss)}")


#  For One-Hot Encoded Class Targets.
print("\nFor One-Hot Encoded Class Targets")
class_targets = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0]
])

correct_confidences = np.sum(softmax_output * class_targets, axis=1)
loss = -np.log(correct_confidences)

print(f"Loss = {loss}")

print(f"Average Loss = {np.mean(loss)}")
