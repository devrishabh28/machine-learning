import numpy as np

weights = np.array([
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
])

#  Vanilla python

dL1 = []  # Array of partial derivatives of L1 regularization
for neuron in weights:
    neuron_dL1 = []  # Derivatives related to one neuron
    for weight in neuron:
        if weight >= 0:
            neuron_dL1.append(1)
        else:
            neuron_dL1.append(-1)
    dL1.append(neuron_dL1)

print(dL1)

#  Using numpy
dL1 = np.ones_like(weights)
dL1[weights < 0] = -1
print(dL1)
