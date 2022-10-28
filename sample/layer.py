import numpy as np

from typing import Callable

class Layer:
    def __init__(self, in_nodes: int, out_nodes: int, 
                 weights: np.array, biases: np.array):
        self.weights = np.full((in_nodes, out_nodes), 1) if weights is None else weights
        self.bias = np.full((in_nodes), 1) if biases is None else biases

    def run(self, inputs: np.array, activation_func: Callable[[float], float]):
        out_values = []
        for node, node_weight, node_bias in zip(inputs, self.weights, self.bias):
            out_node = activation_func(sum(node*node_weight) + node_bias)
            out_values.append(out_node)
        return np.asarray(out_values)

if __name__ == "__main__":
    layer = Layer(10, 5)
    print(layer.weights, layer.bias)
