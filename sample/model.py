from contextlib import contextmanager
from enum import Enum
import os

from model_file_formatter import file_reader, str_writer

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

    def load_wnb(self):
        if not self._path: return None, None
        with open(self._path) as file:
            return file_reader(file)

    def save_wnb(self, weights, biases):
        with open(self._path, "w") as file:
            formatted_str = str_writer(weights, biases)
            file.write(formatted_str)


if __name__ == "__main__":
    if Model.model1.have_wnb:
        weights, biases = Model.model1.load_wnb()
        print(f"weights: {weights[0]}, biases: {biases[0]}")
        Model.model1.save_wnb(weights, biases)