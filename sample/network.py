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
        # Soft-max function on output values
        exp_out = np.exp(inputs)
        layer_sum = np.sum(exp_out)
        confidence_scores = exp_out / layer_sum
        return confidence_scores

    def calculate_loss(self, data: Iterable) -> float:
        batch_outputs = np.array([])
        batch_targets = np.array([])
        for sample in data:
            batch_outputs.append(self.execute(sample[0]))
            batch_targets.append(sample[1])
        
        batch_loss = self.model.loss_func.forward(batch_outputs, batch_targets)
        return batch_loss
        
    def train(self, data: Iterable):
        """
        Tweaks the weight and bias values in the network\n
        data: should be a iteratable yielding tuples containing:\n
            1. a flatten np.ndarray of image data
            2. a np.ndarray of zeros, containng one "1", representing the label after the datasets mapping
        """

        back_input = None # Loss backpropagation
        for layer in self.layers[::-1]:
            for i, v in enumerate(back_input.copy()):
                back_input[i] = self.model.activation_func.df(v) * v
            back_input = layer.back(back_input, self.model.learn_rate)



if __name__ == "__main__":
    network = Network(model.Model.test_model)
    dataset = data.CompiledDataset("emnist-letters.mat", validation_partition=True, 
                                   as_array=True, flatten=True)
    tdata = next(dataset.next_batch(1))
    print(network.execute(tdata[0]), tdata[1])
    print("-------------- Training ----------------")
    network.train(dataset.next_batch(1000))
    tdata = next(dataset.next_batch(1))
    print(network.execute(tdata[0]), tdata[1])

    # network.save_wnb()