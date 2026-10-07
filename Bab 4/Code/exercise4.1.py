# Exercise 4.1 Single Neuron Performing Logical OR
from numpy import exp, array, random, dot, hstack, ones

# Set up parameters ===============================================
# Truth table of logical OR
inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = array([[0, 1, 1, 1]]).T

# Add a bias input (always 1) so the neuron can output 0 for input [0, 0]
inputs_b = hstack((inputs, ones((inputs.shape[0], 1))))
random.seed(1)
weights = 2 * random.random((3, 1)) - 1


# Define a Single Neuron function =================================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output


# Train the Neuron ================================================
for iteration in range(50000):
    output = neuron(inputs_b, weights)
    weights += dot(inputs_b.T, (outputs - output) * output * (1 - output))

# Test the Neuron =================================================
print("Logical OR")
for x in inputs:
    y = neuron(array([x[0], x[1], 1]), weights)
    print(x, "->", round(float(y[0]), 4), "->", int(y[0] > 0.5))
