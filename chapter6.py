import numpy as np

class ActivationFunctions:
    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)
    
    @staticmethod
    def relu_derivative(Z):
        return Z > 0

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))
    
    @staticmethod
    def sigmoid_derivative(Z):
        sig = 1 / (1 + np.exp(-Z))
        return sig * (1 - sig)

    @staticmethod
    def tanh(Z):
        return np.tanh(Z)
    
    @staticmethod
    def tanh_derivative(Z):
        return 1 - np.tanh(Z) ** 2
    
    @staticmethod
    def softmax(Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / expZ.sum(axis=0, keepdims=True)
    
class DeepNeuralNetwork:
    def __init__(self, layers_dims): #sample [5,4,3,2,1] number of neutrons in each layer and the length of the array is number of layer.
        self.parameters = {}
        self.L = len(layers_dims) - 1  # Number of layers (excluding input layer)
        self.initialize_parameters(layers_dims)
    
    def initialize_parameters(self, layers_dims):
        np.random.seed(1)
        for l in range(1, len(layers_dims)):
            self.parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    
    def forward_propagation(self, X):
        A = X
        cache = {}
        
        for l in range(1, self.L):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A) + b
            A = ActivationFunctions.relu(Z)
            cache['A' + str(l)] = A
            cache['Z' + str(l)] = Z
            
        # Output layer: using softmax for multiclass classification
        W = self.parameters['W' + str(self.L)]
        b = self.parameters['b' + str(self.L)]
        Z = np.dot(W, A) + b
        A = ActivationFunctions.softmax(Z)
        cache['A' + str(self.L)] = A
        cache['Z' + str(self.L)] = Z
        
        return A, cache
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(np.multiply(Y, np.log(AL))) / m
        cost = np.squeeze(cost)
        return cost

class DeepNeuralNetwork(DeepNeuralNetwork):
    def backward_propagation(self, X, Y, cache):
        grads = {}
        m = X.shape[1]
        A_prev = cache['A' + str(self.L - 1)]
        A = cache['A' + str(self.L)]
        W = self.parameters['W' + str(self.L)]
        
        dZ = A - Y
        grads['dW' + str(self.L)] = np.dot(dZ, A_prev.T) / m
        grads['db' + str(self.L)] = np.sum(dZ, axis=1, keepdims=True) / m
        
        for l in reversed(range(1, self.L)):
            dA = np.dot(self.parameters['W' + str(l + 1)].T, dZ)
            dZ = dA * ActivationFunctions.relu_derivative(cache['Z' + str(l)])
            A_prev = X if l == 1 else cache['A' + str(l - 1)]
            grads['dW' + str(l)] = np.dot(dZ, A_prev.T) / m
            grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
            
        return grads
    
    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]


class DeepNeuralNetwork(DeepNeuralNetwork):
    def train(self, X, Y, iterations, learning_rate):
        for i in range(iterations):
            # Forward propagation
            AL, cache = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(AL, Y)
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
            
            # Backward propagation
            grads = self.backward_propagation(X, Y, cache)
            
            # Update parameters
            self.update_parameters(grads, learning_rate)