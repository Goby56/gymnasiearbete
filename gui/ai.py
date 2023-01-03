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

def load_model(model_name: str):
    return Model(model_name)
