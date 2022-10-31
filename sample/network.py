import numpy as np

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
        if weights is None:
            for i, nodes in enumerate(self.model.structure[:-1]):
                if weights is None:
                    yield (new_wnb((nodes, self.model.structure[i+1])), new_wnb(nodes))
                else:
                    yield weights[i], biases[i]

    def execute(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.execute(inputs, self.model.activation_func)
        return inputs

    @staticmethod
    def __node_loss(out: float, expected: float) -> float:
        error = out - expected
        return error * error

    def loss(self, data_in: np.ndarray, data_out: np.ndarray) -> float:
        layer_loss = map(self.__node_loss, self.execute(data_in), data_out)
        return sum(layer_loss)

if __name__ == "__main__":
    network = Network(model.Model.test_model)
    dataset = data.CompiledDataset("emnist-letters.mat", validation_partition=True, 
                                   as_array=True, flatten=True)
    costs = [network.loss(*_data) for _data in dataset.next_batch(10)]
    print(costs)

    # weights, biases = [], []
    # for layer in network.layers:
    #     weights.append(layer.weights)
    #     biases.append(layer.bias)
    # network.model.save_wnb(weights, biases)