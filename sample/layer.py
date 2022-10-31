import numpy as np
from typing import Callable

import model

class Layer:
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights
        self.bias = biases

    def execute(self, inputs: np.ndarray, activation_func: model.Activation) -> np.ndarray:
        out_values = []
        for node, node_weight, node_bias in zip(inputs, self.weights, self.bias):
            out_node = activation_func(sum(node*node_weight) + node_bias)
            out_values.append(out_node)
        return np.asarray(out_values)


if __name__ == "__main__":
    print("haha")