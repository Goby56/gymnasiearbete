import numpy as np

from typing import Callable

def _new_wnb(*shape: int) -> np.ndarray:
    return np.random.uniform(low=0, high=1, size=shape)

class Layer:
    def __init__(self, in_nodes: int, out_nodes: int, 
                 weights: np.ndarray, biases: np.ndarray):
        self.weights = _new_wnb(in_nodes, out_nodes) if weights is None else weights
        self.bias = _new_wnb(in_nodes) if biases is None else biases

    def run(self, inputs: np.ndarray, activation_func: Callable[[float], float]): # gör om den är bahd
        out_values = []
        for node, node_weight, node_bias in zip(inputs, self.weights, self.bias):
            out_node = activation_func(sum(node*node_weight) + node_bias)
            out_values.append(out_node)
        return np.asarray(out_values)

if __name__ == "__main__":
    # layer = Layer(10, 5)
    # print(layer.weights, layer.bias)
    print(_new_wnb(4))
