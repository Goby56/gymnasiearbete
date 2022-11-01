import numpy as np
from typing import Generator, Iterable

import layer, model, data

class Network:
    def __init__(self, model: model.Model):
        self.model = model
        self.layers = []
        for w, b in self.__load_wnb():
            self.layers.append(layer.Layer(w, b))

    def __load_wnb(self):
        """
        Loads model .wnb file 
        Creates wnb for network (not file) if .wnb is missing
        """
        new_wnb = lambda shape: np.random.uniform(low=0, high=1, size=shape)
        weights, biases = self.model.load_wnb()
        for i, nodes in enumerate(self.model.structure[:-1]):
            if weights is None:
                yield (new_wnb((nodes, self.model.structure[i+1])), 
                       new_wnb(self.model.structure[i+1])) 
            else:
                yield (weights[i], biases[i])

    def save_wnb(self):
        w, b = [], []
        for layer in self.layers:
            w.append(layer.weights)
            b.append(layer.bias)
        self.model.save_wnb(w, b)

    def execute(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.execute(inputs, self.model.activation_func)
        return inputs

    @staticmethod
    def __node_loss(out: float, expected: float) -> float:
        error = out - expected
        return error * error

    def loss(self, data_in: np.ndarray, data_out: np.ndarray) -> float:
        """
        Find the total network loss for one "image".
        data_in: np.ndarray should be a flattend array containing the image data
        data_out: np.ndarray should be an array of zeros, conatining only one "1" at the spot
        corresponding with it's label acording to the dataset mapping.

        returns float
        """
        layer_loss = map(self.__node_loss, self.execute(data_in), data_out)
        return sum(layer_loss)

    def intervall_loss(self, data: Iterable) -> float:
        cost = 0
        for _data in data:
            cost += self.loss(*_data)
        return cost

    def train(self, data: Iterable):
        """
        Tweaks the weight and bias values in the network\n
        data: should be a iteratable yielding tuples containing:\n
            1. a flatten np.ndarray of image data
            2. a np.ndarray of zeros, containng one "1", representing the label after the datasets mapping
        """
        h = 0.0001
        cost = self.intervall_loss(data)

        for layer in self.layers:
            weight_gradient = np.empty(layer.weights.shape)
            bias_gradient = np.empty(layer.bias.shape)
            for out in range(layer.out_nodes):
                for _in in range(layer.in_nodes):
                    layer.weights[_in, out] += h
                    dc = self.intervall_loss(data) - cost
                    layer.weights[_in, out] -= h
                    weight_gradient[_in, out] = dc / h
                
                layer.bias[out] += h
                dc = self.intervall_loss(data) - cost
                layer.bias[out] -= h
                bias_gradient[out] = dc / h

            layer.bias -= bias_gradient * self.model.learn_rate
            layer.weights -= weight_gradient * self.model.learn_rate


if __name__ == "__main__":
    network = Network(model.Model.test_model)
    dataset = data.CompiledDataset("emnist-letters.mat", validation_partition=True, 
                                   as_array=True, flatten=True)
    tdata = next(dataset.next_batch(1))
    print(network.execute(tdata[0]), tdata[1])
    print("-------------- training ----------------")
    network.train(dataset.next_batch(1000))
    tdata = next(dataset.next_batch(1))
    print(network.execute(tdata[0]), tdata[1])

    # network.save_wnb()