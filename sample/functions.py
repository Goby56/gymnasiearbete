from enum import Enum
from typing import Callable

import math
import numpy as np

# region Activation

class Activation_Softmax:
    """
    Softmax funtion is always applied as the last activation function in a network
    The derivitive is included in Loss_CCE backward calculation
    """
    def forward(self, inputs):
        """
        takes an array as argument
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        return probabilities

    def backward(self, x):
        raise NotImplementedError

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        inputs = dvalues.copy()
        inputs[self.inputs <= 0] = 0
        return inputs

class Activation_ReLU6:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, np.minimum(inputs, 6))

    def backward(self, dvalues):
        inputs = dvalues.copy()
        inputs[np.logical_and(6 <= self.inputs, self.inputs >= 0)] = 0
        return inputs


class Activation_Sigmoid:
    # TODO: prob. fix 

    def forward(self, X):
        self.inputs = X
        return 1 / (1 + np.exp(-self.inputs))

    def backward(self, X):
        return self.inputs * self.forward(X) * (1- self.forward(X))

class Activation_Step:
    def f(self, x):
        return x > 0

    def df(self, x):
        return 0

# endregion Activation

# region Loss

class Loss:
    def __init__(self):
        self.reset_accumulation()

    def reset_accumulation(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    def calculate(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        batch_loss = self.forward(batch_outputs, batch_targets)
        # print(batch_loss)
        # print(np.mean(batch_loss))
        self.accumulated_sum += np.sum(batch_loss)
        self.accumulated_count += len(batch_loss)
        return np.mean(batch_loss)

    def calculate_accumulated(self):
        accumulated_loss = self.accumulated_sum / self.accumulated_count
        self.reset_accumulation()
        return accumulated_loss
        

class Loss_CCE(Loss):
    """
    Forward and backward pass of the Categorical Cross-Entropy loss function. Backward pass
    also includes the derivative of the Soft-Max function.
    """
    def forward(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        batch_outputs = np.clip(batch_outputs, 1e-7, 1-1e-7)
        confidences = np.sum(batch_outputs*batch_targets, axis=1)
        losses = -np.log(confidences)
        return losses

    def backward(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        num_of_samples = len(batch_outputs)
        backprop_input = batch_outputs.copy()
        backprop_input[range(num_of_samples), np.argmax(batch_targets, axis=1)] -= 1
        return backprop_input / num_of_samples

# endregion Loss

# region Accuracy

class Accuracy:
    def __init__(self):
        self.reset_accumulation()

    def reset_accumulation(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        comparisons = self.compare(batch_outputs, batch_targets)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        accumulated_accuracy = self.accumulated_sum / self.accumulated_count
        self.reset_accumulation()
        return accumulated_accuracy

class Accuracy_Categorical(Accuracy):

    def compare(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        predictions = np.argmax(batch_outputs, axis=1)
        if len(batch_targets.shape) == 2:
            batch_targets = np.argmax(batch_targets, axis=1)
        return predictions == batch_targets

# endregion Accuracy

# region Optimizer

class Optimizer_SGD:
    """
    Stochastic gradient decent - the most simple optimizer algorithm
    """
    def __init__(self, learn_rate, decay):
        self.learn_rate = learn_rate
        self.current_learn_rate = learn_rate
        self.decay = decay
        self.iterations = 0

    def apply_decay(self):
        if self.decay > 0:
            self.current_learn_rate = self.learn_rate * (1 / (1 + self.decay * self.iterations))
        self.iterations += 1

    def apply_training(self, layer):
        layer.weights -= self.learn_rate * layer.dweights
        layer.biases -= self.learn_rate * layer.dbiases

# endregion Optimizer