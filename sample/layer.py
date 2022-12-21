import numpy as np
from typing import Callable

import model, logger

class Layer:
    id = 0
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights
        self.bias = biases
        self.in_nodes, self.out_nodes = weights.shape
        self.id = Layer.id
        Layer.id += 1

        self.dweights = np.zeros(weights.shape)
        self.dbias = np.zeros(biases.shape)

    #@logger.log_function()
    def execute(self, inputs: np.ndarray, activation_func: model.Activation) -> np.ndarray:
        """
        forward pass through the network
        """
        # TODO: fuckar nätverket. fixa. nice!
        raise Exception("HÄR ÄR ETT FEL. DET MÅSTE FIXAS. EFTER FÖRSTA GÅNGEN DEN KÖRS RETURNAS BARA ETTOR, AKTIVERINGSFUNTIONEN GÖR SÅ ALLA TAL BLIR ETT REEEEE REEEE RAHHH")
        self.inputs = inputs
        output = np.dot(inputs, self.weights) + self.bias # Fig. 2
        
        with np.nditer(output, op_flags=["readwrite"]) as it: # Iterate and modify all elements
            for value in it: value[...] = activation_func(value)
        return output

    def back(self, dvalues: np.ndarray, learn_rate: float) -> np.ndarray:
        """
        backward pass through the network
        """
        self.dweights -= learn_rate * np.dot(self.inputs.T, dvalues)
        self.dbias -= learn_rate * np.sum(dvalues, axis=0)
        return np.dot(dvalues, self.weights.T)

    def apply_trainings(self) -> None:
        self.weights = self.dweights
        self.bias = self.dbias

if __name__ == "__main__":
    print("Du kör fel fil din dummer! :(")