import numpy as np

class SoftmaxActivation:
    
    def forward(self, inputs):
        
       
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        
       
        exp_values = np.exp(shifted_inputs)
        
        
        sum_exp_values = np.sum(exp_values, axis=1, keepdims=True)
        
        
        probabilities = exp_values / sum_exp_values
        
        return probabilities


softmax = SoftmaxActivation()


inputs = np.array([[2.0, 1.0, 0.1],
                   [1.0, 3.0, 0.5]])


output = softmax.forward(inputs)
print("Softmax probabilities:\n", output)