#  An example code for the dropout layer.
import random
import numpy as np

dropout_rate = 0.5
#  Example output containing 10 values
example_output = [0.27, -1.03, 0.67, 0.99,
                  0.05, -0.37, -2.01, 1, 13, -0.07, 0.73]


#  Vanilla python

#  Repeat as long as neccessary
while True:

    #  Randomly choose index and set value to 0
    index = random.randint(0, len(example_output) - 1)
    example_output[index] = 0

    dropped_out = 0
    for value in example_output:
        if value == 0:
            dropped_out += 1

    #  If required number of outputs is zeroed - leave the loop

    if dropped_out / len(example_output) >= dropout_rate:
        break

print(example_output)


#  Using numpy
dropout_rate = 0.3
example_output = np.array(example_output)

example_output *= np.random.binomial(1, 1-dropout_rate, example_output.shape)
print(example_output)


#  Scaling

dropout_rate = 0.2
example_output = np.array([0.27, -1.03, 0.67, 0.99,
                          0.05, -0.37, -2.01, 1, 13, -0.07, 0.73])

print(f'sum initial {sum(example_output)}')

sums = []

for _ in range(10001):

    example_output2 = example_output * \
        np.random.binomial(1, 1 - dropout_rate,
                           example_output.shape) / (1 - dropout_rate)
    sums.append(example_output2.sum())


print(f"mean sum: {np.mean(sums)}")
