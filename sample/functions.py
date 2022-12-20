from enum import Enum
from typing import Callable

import math
import numpy as np


class Activation(Enum):
    ReLU = (lambda x: max(0, x), lambda x: x > 0)
    Sigmoid = (lambda x: 1 / (1 + np.exp(-x)),
               lambda x: (1 / (1 + np.exp(-x))) * (1-(1 / (1 + np.exp(-x)))))
    Step = (lambda x: x > 0, lambda x: 0)

    SiLU = lambda x: x / (1 + np.exp(-x)) # deprecated for now

    def __init__(self, func: Callable, derivative: Callable):
        self.__f = func
        self.__df = derivative

    def f(self, __x: float) -> float:
        return float(self.__f(__x))

    def df(self, __x: float) -> float:
        return float(self.__df(__x))

    def __call__(self, __x: float) -> float:
        return self.f(__x)

# region Loss

class Loss:
    def forward(batch_outputs: np.ndarray, batch_targets: np.ndarray) -> float:
        raise NotImplementedError

    def backward(batch_outputs: np.ndarray, batch_targets: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Loss_CCE(Loss): 
    """
    Forward and backward pass of the Categorical Cross-Entropy loss function. Backward pass
    also calculates the derivative of the Soft-Max function.
    """
    def forward(batch_outputs: np.ndarray, batch_targets: np.ndarray):
        batch_outputs = np.clip(batch_outputs, 1e-7, 1-1e-7)
        confidences = np.sum(batch_outputs*batch_targets, axis=1)
        losses = -np.log(confidences)
        return np.mean(losses)

    def backward(batch_outputs: np.ndarray, batch_targets: np.ndarray):
        num_of_samples = len(batch_outputs)
        backprop_input = batch_outputs.copy()
        backprop_input[range(num_of_samples), np.argmax(batch_targets, axis=1)] -= 1
        return backprop_input / num_of_samples

# endregion Loss

class Accuracy:
    @staticmethod
    def calc(batch_outputs: np.ndarray, batch_targets: np.ndarray):
        predictions = np.argmax(batch_outputs, axis=1)
        if len(batch_targets.shape) == 2:
            batch_targets = np.argmax(batch_targets, axis=1)
        accuracy = np.mean(predictions == batch_targets)
        return accuracy