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
        # Normalization using Soft-Max function
        exp_out = np.exp(inputs)
        layer_sum = np.sum(exp_out)
        confidence_scores = exp_out / layer_sum
        return confidence_scores

    def run_batch(self, data: Iterable) -> float:
        batch_outputs = []
        batch_targets = []
        for sample in data:
            batch_outputs.append(self.execute(sample[0]))
            batch_targets.append(sample[1])
        return np.array(batch_outputs), np.array(batch_targets)

    def train(self, data: Iterable):
        batch_outputs, batch_targets = self.run_batch(data)
        batch_loss = self.model.loss_func.forward(batch_outputs, batch_targets)

        # TODO fix batch backprop problem (matrix is being inputed instead of vector)
        batches = self.model.loss_func.backward(batch_outputs, batch_targets)# Loss backpropagation
        backprop_input = batches[0]
        for batch in batches:
            backprop_input += batch
        backprop_input /= 10
        
        print("Loss:", batch_loss)

        for layer in self.layers[::-1]:
            for i, v in enumerate(backprop_input.copy()):
                backprop_input[i] = self.model.activation_func.df(v) * v
            backprop_input = layer.back(backprop_input, self.model.learn_rate)
            # print(layer.weights)


if __name__ == "__main__":
    network = Network(model.Model.test_model)
    dataset = data.CompiledDataset("emnist-letters.mat", validation_partition=True, 
                                   as_array=True, flatten=True, normalize=True)
    print("-------------- Training ----------------")
    for _ in range(1000):
        network.train(dataset.next_batch(10))
    tdata = next(dataset.next_batch(1))
    print(network.execute(tdata[0]), tdata[1])

    # network.save_wnb()