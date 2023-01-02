import numpy as np
import itertools

from . import model, layer, utils, data

# TODO: Move standardization from dataset to network. option has been added to config.
class Network:
    def __init__(self, model: model.Model):
        self.model = model

        if self.model.has_wnb:
            # Load weights and biases from .wnb file
            weights, biases = self.model.load_wnb()
        else:
            # Initialize weights and biases
            weights = list(map(self.init_weights,
                utils.pairwise(self.model.structure["nodes"])))
            biases = list(map(self.init_biases, 
                self.model.structure["nodes"][1:]))

        # Create layers
        self.layers = [layer.Layer(w, b, a()) for w, b, a in 
            zip(weights, biases, self.model.structure["activations"])]

    def init_weights(self, size):
        return 0.01 * np.random.randn(*size)

    def init_biases(self, size):
        return np.zeros(shape=size)

    def forward(self, samples: np.ndarray):
        output = self.layers[0].forward(samples)
        for layer in self.layers[1:]:
            output = layer.forward(output)
        
        return output

    def backward(self, samples, labels):
        loss_gradients = self.model.loss_function.backward(samples, labels)
        gradients = self.layers[-1].backward(loss_gradients, temp_cce=True)
        for layer in reversed(self.layers[:-1]):
            gradients = layer.backward(gradients, temp_cce=False)

        self.model.optimizer.apply_decay()
        for layer in self.layers:
            self.model.optimizer.apply_training(layer)
    
    def train(self, data_points):
        samples, labels = map(np.asarray, zip(*data_points))
        guesses = self.forward(samples)
        
        loss = self.model.loss_function.calculate(guesses, labels)
        accuracy = self.model.accuracy_function.calculate(guesses, labels)

        self.backward(guesses, labels)

        return loss, accuracy

    def save(self):
        w, b = zip(*[(i.weights, i.biases) for i in self.layers])
        self.model.save_wnb(w, b)

if __name__ == "__main__":
    print("fel fel fel feil")



