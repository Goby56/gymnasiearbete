import layer
import model
from typing import Callable

class Network:
    def __init__(self, model: model.Model):
        self.model = model
        self.layers = []
        w, b = self.model.load_wnb()
        for i in range(len(self.model.structure)-1):
            layer.append(layer.Layer(self.model.structure[i], 
                                     self.model.structure[i+1], 
                                     w, b))
