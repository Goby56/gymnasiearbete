import numpy as np
import enum, itertools
from typing import Generator, Iterable
from functions import Accuracy
import layer, model, data, logger, utils

class Mode(enum.Enum):
    debug = enum.auto()
    train = enum.auto()
    normal = enum.auto()

class Network:
    def __init__(self, model: model.Model, mode=Mode.normal):
        self.model = model
        self.__load_layers()
        self.mode = mode

    def __load_layers(self):
        """
        Loads layers via model
        If model has .wnb file, it is loaded
        If not it creates random weights and biases between -1, 1
        """

        if self.model.has_wnb:
            weights, biases = self.model.load_wnb()
        else:
            weights = list(map(lambda x: np.random.uniform(low=-1, high=1, size=x),
                utils.pairwise(self.model.structure)))
            biases = list(map(lambda x: np.random.uniform(low=-1, high=1, size=x),
                self.model.structure[1:]))

        self.layers = list(itertools.starmap(layer.Layer, zip(weights, biases)))

    def save_wnb(self):
        w, b = zip(*[(i.weights, i.bias) for i in self.layers])
        self.model.save_wnb(w, b)

    def execute(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.execute(inputs, self.model.activation_function.f)

        # Normalization using Soft-Max function
        exp_out = np.exp(inputs)
        layer_sum = np.sum(exp_out)
        confidence_scores = exp_out / layer_sum # skapar ett error om man har ReLU som aktiveringsfunktionen
        return confidence_scores

    def run_batch(self, data: Iterable) -> tuple:
        samples, labels = map(np.asarray, zip(*data))
        return self.execute(samples), labels

    def back_propagate(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        # Loss backpropagation
        bp_value = self.model.loss_function.backward(batch_outputs, batch_targets)
        # backwardpass through the layers
        for layer in reversed(self.layers):
            # Iterate and modify all elements
            with np.nditer(bp_value, op_flags=["readwrite"]) as it:
                # f'(x) * x for every element in matrix
                for v in it: v[...] = self.model.activation_function.df(v) * v
            bp_value = layer.back(bp_value, self.model.learn_rate)
        # applies dweights and dbiases
        
        for layer in self.layers:
            layer.apply_trainings()
        
        # TODO ADD OPTIMIZERS SUCH AS ADAM (BAD) AND ADAGRAD

    def train(self, data: Iterable):
        batch_outputs, batch_targets = self.run_batch(data)

        # TODO ADD DATA_LOSS AND L1 + L2 REGULARIZATION LOSS
        loss = self.model.loss_function.calculate(batch_outputs, batch_targets)
        accuracy = self.model.accuracy_function.calculate(batch_outputs, batch_targets)

        self.back_propagate(batch_outputs, batch_targets)

        NOT_IMPLEMENTED = 0

        return accuracy, loss, NOT_IMPLEMENTED, NOT_IMPLEMENTED, NOT_IMPLEMENTED

    def _debug_get_weights(self):
        return np.array([layer.weights for layer in self.layers], dtype=object)

if __name__ == "__main__":
    model = model.Model("test_model")
    network = Network(model)
    dataset = data.CompiledDataset(
        "emnist-letters.mat", 
        validation_partition=True, 
        as_array=True, 
        flatten=True, 
        normalize=True
    )
    
    start_weights = network._debug_get_weights()
    # -------- training -----------
    for _ in range(10):
        network.train(dataset.next_batch(10))
        ree = network._debug_get_weights()
        print("DIFF", ree - start_weights, "\nWIEGHTG", ree)
        start_weights = ree

    logger.print_post()
    # network.save_wnb()