import numpy as np


class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = np.array(weights)
        self.threshold = threshold

 
    def activation(self, weighted_sum):
        if weighted_sum >= self.threshold:
            return 1
        else:
            return 0

    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return self.activation(weighted_sum)

inputs = np.array([1, 0, 1])  
weights = np.array([1, 1, -1])  
threshold = 1  

neuron = McCullochPittsNeuron(weights, threshold)


output = neuron.forward(inputs)
print(f"Neuron output for inputs {inputs}: {output}")