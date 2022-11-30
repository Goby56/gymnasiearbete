import numpy as np
from typing import Callable

import model

class Layer:
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights
        self.bias = biases
        self.in_nodes, self.out_nodes = weights.shape

    def execute(self, inputs: np.ndarray, activation_func: model.Activation) -> np.ndarray:
        a = np.dot(inputs, self.weights) + self.bias # Fig. 2
        for i, b in enumerate(a):
            a[i] = activation_func(b)
        return a