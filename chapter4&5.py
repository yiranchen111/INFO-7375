import numpy as np

class Activation:

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))
    
    @staticmethod
    def sigmoid_derivative(A):
        return A * (1 - A)
    
class Neuron:
    def __init__(self, input_size, activation_function):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0
        self.activation_function = activation_function

    def forward(self, inputs):
        self.z = np.dot(inputs, self.weights) + self.bias
        self.a = self.activation_function(self.z)
        return self.a

class Layer:
    def __init__(self, num_neurons, input_size, activation_function):
        self.neurons = [Neuron(input_size, activation_function) for _ in range(num_neurons)]

    def forward(self, inputs):
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons]).T
        return self.outputs

class Model:
    def __init__(self):
        self.hidden_layer = None
        self.output_layer = None
        self.activations = []  

    def add_hidden_layer(self, layer):
        self.hidden_layer = layer

    def add_output_layer(self, layer):
        self.output_layer = layer

    def forward_propagation(self, X):
        self.activations = [X]  
        
        X = self.hidden_layer.forward(X)
        self.activations.append(X)  
        
        X = self.output_layer.forward(X)
        self.activations.append(X)  
        
        return X
    
class LossFunction:
    @staticmethod
    def cross_entropy_loss(y_true , y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred , epsilon , 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def cross_entropy_loss_derivative(y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

class ForwardProp:
    def __init__(self, model):
        self.model = model

    def compute(self, X):
        return self.model.forward_propagation(X)


class BackProp:
    def __init__(self, model, loss_function):
        self.model = model
        self.loss_function = loss_function

    def compute(self, X, y_true):
        y_pred = self.model.forward_propagation(X)

        gradients = {
            'dW_hidden': None,
            'dB_hidden': None,
            'dW_output': None,
            'dB_output': None
        }

      
        delta_output = self.loss_function.cross_entropy_loss_derivative(y_true, y_pred)
        
       
        activation_hidden = self.model.activations[1]
        
        
        dW_output = np.dot(delta_output.T, activation_hidden) / y_true.shape[0]
        dB_output = np.sum(delta_output, axis=0) / y_true.shape[0]

        
        delta_hidden = np.dot(delta_output, np.array([neuron.weights for neuron in self.model.output_layer.neurons])) * Activation.sigmoid_derivative(activation_hidden)
        
        activation_input = self.model.activations[0]  
        
       
        dW_hidden = np.dot(delta_hidden.T, activation_input) / y_true.shape[0]
        dB_hidden = np.sum(delta_hidden, axis=0) / y_true.shape[0]

        gradients['dW_hidden'] = dW_hidden
        gradients['dB_hidden'] = dB_hidden
        gradients['dW_output'] = dW_output
        gradients['dB_output'] = dB_output

        return gradients
    
class GradDescent:
    @staticmethod
    def update(weights, biases, dW, dB, learning_rate):
        weights -= learning_rate * dW
        biases -= learning_rate * dB


class Training:
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.backprop = BackProp(model, loss_function)

    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.model.forward_propagation(X_train)
            
            loss = self.loss_function.cross_entropy_loss(y_train, y_pred)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

            gradients = self.backprop.compute(X_train, y_train)

           
            for i in range(len(self.model.hidden_layer.neurons)):
                self.optimizer.update(self.model.hidden_layer.neurons[i].weights, 
                                      self.model.hidden_layer.neurons[i].bias, 
                                      gradients['dW_hidden'][i], gradients['dB_hidden'][i], learning_rate)  

        
            for i in range(len(self.model.output_layer.neurons)):
                self.optimizer.update(self.model.output_layer.neurons[i].weights, 
                                      self.model.output_layer.neurons[i].bias, 
                                      gradients['dW_output'][i], gradients['dB_output'][i], learning_rate)
                

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])


model = Model()

hidden_layer = Layer(num_neurons=4, input_size=2, activation_function=Activation.sigmoid)
model.add_hidden_layer(hidden_layer)


output_layer = Layer(num_neurons=1, input_size=4, activation_function=Activation.sigmoid)
model.add_output_layer(output_layer)


loss_function = LossFunction()
optimizer = GradDescent()


training = Training(model, loss_function, optimizer)
training.train(X_train, y_train, epochs=1000, learning_rate=0.01)