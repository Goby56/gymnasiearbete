import os, json

try:
    from . import functions
    from .model_file_formatter import file_reader, str_writer
except ImportError:
    import functions
    from model_file_formatter import file_reader, str_writer

_MODEL_FOLDER = os.path.join(os.path.dirname( __file__ ), "..", "data\\models")

class MissingConfigAttribute(Exception):
    """Expetion raised when config.json is missing required options."""

    def __init__(self, attr):
        self.attr = attr
        super().__init__(f"config.json is missing required attribute '{attr}'")

class Model(dict):

    function_bindings = {
        "loss_function": "Loss_",
        "accuracy_function": "Accuracy_",
        "optimizer": "Optimizer_"
    }

    def __init__(self, name: str):
        self.name = name

        __dir_path = os.path.join(_MODEL_FOLDER, self.name)
        self.paths = {
            "dir": __dir_path,
            "config": os.path.join(__dir_path, "config.json"),
            "wnb": os.path.join(__dir_path, self.name + ".wnb"),
        }

        self.has_wnb = os.path.exists(self.paths["wnb"])

        with open(self.paths["config"]) as file:
            config = json.load(file)
            setup = config["setup"]
            training = config["training"]
            
        for key in {*setup.keys()} & {*training.keys()}:
            assert setup[key] == training[key]

        super().__init__(setup | training)

        def get(var):
            function = getattr(functions, var, None)
            if function is None:
                raise Exception(f'function "{function}" does not exist!')
            return function

        # init setup hyperparams
        for i, func in enumerate(self.structure["activations"].copy()):
            self.structure["activations"][i] = get("Activation_" + func)

        # init training hyperparams
        for func in Model.function_bindings:
            if func == "optimizer":
                binding = Model.function_bindings[func] + self[func]["function"]
                optimizer = get(binding)
                self[func] = optimizer(**self[func]["args"])
            else:
                self[func] = get(Model.function_bindings[func] + self[func])()

        in_, *_, out_ = self.structure["nodes"]
        self.shape = (in_, out_)

    def __getattr__(self, attr):
        return self[attr]

    def __missing__(self, attr):
        raise MissingConfigAttribute(attr)

    def load_wnb(self):
        """
        Loads weights and bias from file {model name}.wnb
        returns (ndarray, ndarray)
        """
        with open(self.paths["wnb"]) as file:
            return file_reader(file)

    def save_wnb(self, weights, biases):
        """
        Saves weights and bias to file {model name}.wnb
        Creates file if it doesn't exist

        weights: an array or list of each layers weights
        biases: an array or list of each layers biases
        """
        with open(self.paths["wnb"], "w") as file:
            formatted_str = str_writer(weights, biases)
            file.write(formatted_str)


if __name__ == "__main__":
    import numpy as np
    model = Model("test_model")
    r = model.structure["activations"][0]().forward(np.array([10, -1, 6, 2, 1]))
    print(r)