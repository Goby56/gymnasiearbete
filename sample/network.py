import numpy as np
from typing import Generator, Iterable
from functions import Accuracy
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

    def run_batch(self, data: Iterable) -> tuple:
        # TODO: fix so this is a nice one line np trick
        samples, labels = [], []
        for sample, label in data:
            samples.append(sample)
            labels.append(label)
        samples, labels = np.array(samples), np.array(labels)
        return self.execute(samples), labels

    def train(self, data: Iterable):
        batch_outputs, batch_targets = self.run_batch(data)
        batch_loss = self.model.loss_func.forward(batch_outputs, batch_targets)
        batch_acc = Accuracy.calc(batch_outputs, batch_targets)

        # TODO fix batch backprop problem (matrix is being inputed instead of vector)
        bp_value = self.model.loss_func.backward(batch_outputs, batch_targets)# Loss backpropagation
        
        print(f"Loss: {batch_loss:.2f}, Accuracy: {batch_acc:.2f}")

        for layer in self.layers[::-1]:
            with np.nditer(bp_value.copy(), op_flags=["readwrite"]) as it: # Iterate and modify all elements
                for v in it: v[...] = self.model.activation_func.df(v) * v # Derivative
            bp_value = layer.back(bp_value, self.model.learn_rate)
             
        for layer in self.layers:
            layer.apply_trainings()


if __name__ == "__main__":
    network = Network(model.Model.test_model)
    dataset = data.CompiledDataset("emnist-letters.mat", validation_partition=True, 
                                   as_array=True, flatten=True, normalize=True)
    
    # tdata = dataset.next_batch(10)
    # results, labels = network.run_batch(tdata)
    # pairs = zip(results, labels)
    # print(list(pairs)[0])

    # print("-------------- Training ----------------")
    for _ in range(1000):
        network.train(dataset.next_batch(10))

    # delta = network.execute(tdata[0]) - result
    # print(delta)

    # network.save_wnb()