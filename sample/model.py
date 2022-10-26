from enum import Enum
from typing import Callable
import os

_MODEL_FOLDER = ""

class Activation(Enum):
    ReLU = lambda x: max(0, x)

class Model(Enum):
    model1 = (Activation.ReLU, 1, (2, 3, 2))

    def __init__(self, activation_func: Activation,
                 learn_rate: int, structure: tuple[int]):
        self.activation_func = activation_func
        self.learn_rate = learn_rate
        self.structure = structure

    def load_data(self):
        with open(os.path.join(_MODEL_FOLDER, self._name_)) as file:
            yield file # TODO: FORMAT INFORMATION AND YIELD NP.ARRAYS
