import numpy as np
from typing import Callable

import model

class Layer:
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights
        self.bias = biases
        self.in_nodes, self.out_nodes = weights.shape
        self.inputs = np.empty(self.out_nodes, dtype=float)

    def execute(self, inputs: np.ndarray, activation_func: model.Activation) -> np.ndarray:
        """
        forward pass through the network
        """
        self.inputs = np.dot(inputs, self.weights) + self.bias # Fig. 2
        for i, v in enumerate(self.inputs):
             self.inputs[i] = activation_func(v)
        return self.inputs

    def back(self, dvalues: np.ndarray, learn_rate: float) -> np.ndarray:
        """
        backward pass through the network
        """
        out = np.dot(dvalues, self.weights.T)
        self.bias -= learn_rate * np.sum(dvalues, keepdims=True)
        self.weights -= learn_rate * np.dot(self.inputs, dvalues.T)
        return out