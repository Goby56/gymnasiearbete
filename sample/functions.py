from enum import Enum
from typing import Callable

import math


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