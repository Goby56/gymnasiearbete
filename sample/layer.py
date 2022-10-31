import numpy as np
from typing import Callable

import model

def _new_wnb(*shape: int) -> np.ndarray:
    return np.random.uniform(low=0, high=1, size=shape)

class Layer:
    def __init__(self, in_nodes: int, out_nodes: int, 
                 weights: np.ndarray, biases: np.ndarray):
        self.weights = _new_wnb(in_nodes, out_nodes) if weights is None else weights
        self.bias = _new_wnb(in_nodes) if biases is None else biases

    def execute(self, inputs: np.ndarray, activation_func: model.Activation):
        out_values = []
        for node, node_weight, node_bias in zip(inputs, self.weights, self.bias):
            out_node = activation_func(sum(node*node_weight) + node_bias)
            out_values.append(out_node)
        return np.asarray(out_values)


if __name__ == "__main__":
    w = _new_wnb(2, 3)
    b = _new_wnb(2)
    layer = Layer(2, 3, w, b)
    out1 = layer.execute(np.array([2, 2]), model.Activation.ReLU)
    out2 = layer.execute(np.array([2, 2]), model.Activation.Sigmoid)
    print(out1)
    print(out2)
