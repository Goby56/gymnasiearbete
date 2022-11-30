from enum import Enum
from typing import Callable

import math
import numpy as np


class Activation(Enum):
    ReLU = (lambda x: max(0, x), lambda x: x > 0)
    Sigmoid = (lambda x: 1 / (1 + math.exp(-x)),
               lambda x: (1 / (1 + math.exp(-x))) * (1-(1 / (1 + math.exp(-x)))))
    Step = (lambda x: x > 0, lambda x: 0)

    SiLU = lambda x: x / (1 + math.exp(-x)) # deprecated for now

    def __init__(self, func: Callable, derivative: Callable):
        self.__f = func
        self.__df = derivative

    def f(self, __x: float) -> float:
        return float(self.__f(__x))

    def df(self, __x: float) -> float:
        return float(self.__df(__x))

    def __call__(self, __x: float) -> float:
        return float(self.__f(__x))

class Loss:
    def forward(batch_outputs: np.ndarray, batch_targets: np.ndarray) -> float:
        raise NotImplementedError

    def backward():
        raise NotImplementedError

class Loss_CCE(Loss): 
    """
    Categorical Cross-Entropy loss function
    """
    def forward(batch_outputs: np.ndarray, batch_targets: np.ndarray):
        batch_outputs = np.clip(batch_outputs, 1e-7, 1-1e-7)
        confidences = np.sum(batch_outputs*batch_targets, axis=1)
        losses = -np.log(confidences)
        return np.mean(losses)

    def backward():
        pass