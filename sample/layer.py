import numpy as np

class Layer:
    def __init__(self, weights, biases, activation):
        self.weights = weights
        self.biases = biases
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        output = np.dot(inputs, self.weights) + self.biases # Fig. 2
        return self.activation.forward(output)

    def backward(self, gradients, *, temp_cce):
        # TODO fix independet activation and loss function derivative
        if not temp_cce:
            dvalues = self.activation.backward(gradients)
        else:
            dvalues = gradients

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0)
        return np.dot(dvalues, self.weights.T)
