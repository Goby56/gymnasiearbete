from contextlib import contextmanager
from enum import Enum
import os

from model_file_formatter import file_formatter

_MODEL_FOLDER = os.path.join(os.path.dirname( __file__ ), "..", "data\\models")

class Activation(Enum):
    ReLU = lambda x: max(0, x)

class Model(Enum):
    model1 = (Activation.ReLU, 1, (2, 3, 2))

    def __init__(self, activation_func: Activation,
                 learn_rate: int, structure: tuple[int]):
        self.activation_func = activation_func
        self.learn_rate = learn_rate
        self.structure = structure

        self._path = os.path.join(_MODEL_FOLDER, self._name_ + ".model")
        self.have_wnb = os.path.exists(self._path)

    @contextmanager
    def load_wnb(self):
        with open(self._path) as file:
            yield file_formatter(file)

if __name__ == "__main__":
    if Model.model1.have_wnb:
        with Model.model1.load_wnb() as (weights, biases):
            print(f"weights: {weights[0]}, biases: {biases[0]}")
            