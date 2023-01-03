import sys, os, json
PATH_MODELS = os.path.join(os.getcwd(), "data\\models")

sys.path.append(os.getcwd())
import sample

class Model:
    def __init__(self, model_name):
        self.name = model_name
        self.model = sample.Model(model_name)
        self.network = sample.Network(self.model)
        # TODO: datasets and training?

LOADED_MODELS = []

def load_models():
    global LOADED_MODELS
    LOADED_MODELS = []
    for model_name in os.listdir(PATH_MODELS):
        LOADED_MODELS.append(Model(model_name))

def new_model(model_name: str, config_args: dict):
    model_path = os.path.join(PATH_MODELS, model_name)
    config_path = os.path.join(model_path, "config.json")
    assert not os.path.exists(model_path)
    os.mkdir(model_path)
    with open(config_path, "w") as file:
        json.dump(config_args, file, indent=4)
    
    LOADED_MODELS.append(Model(model_name))