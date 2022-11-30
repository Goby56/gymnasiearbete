from enum import Enum
import os

from functions import Activation
from model_file_formatter import file_reader, str_writer


_MODEL_FOLDER = os.path.join(os.path.dirname( __file__ ), "..", "data\\models")

class Model(Enum):
    model1 = (Activation.ReLU, 1, (2, 3, 2))
    test_model = (Activation.Sigmoid, 0.3, (28*28, 100, 26))

    def __init__(self, activation_func: Activation,
                 learn_rate: int, structure: tuple[int]):
        self.activation_func = activation_func
        self.learn_rate = learn_rate
        self.structure = structure

        self.__path = os.path.join(_MODEL_FOLDER, self._name_ + ".wnb")
        self.have_wnb = os.path.exists(self.__path)

    def load_wnb(self):
        """
        Loads weights and bias from file {enum field name}.wnb
        """
        if not self.have_wnb: return None, None
        with open(self.__path) as file:
            return file_reader(file)

    def save_wnb(self, weights, biases):
        """
        Saves weights and bias to file {enum field name}.wnb
        Creates file if it doesn't exist

        weights: an array or list of each layers weights
        biases: an array or list of each layers biases
        """
        with open(self.__path, "w") as file:
            formatted_str = str_writer(weights, biases)
            file.write(formatted_str)


if __name__ == "__main__":
    if Model.model1.have_wnb:
        weights, biases = Model.test_model.load_wnb()
        print(f"weights: {weights[0]}, biases: {biases[0]}")