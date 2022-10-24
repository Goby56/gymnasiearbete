import layer

from typing import Callable

class Network:
    def __init__(self, activation_func: Callable[[float], float], 
                 learn_rate: int, structure: tuple[int]):
        self.activation_func = activation_func
        self.learn_rate = learn_rate
        
        self.layers = []
        for i in range(len(structure)-1):
            layer.append(layer.Layer(structure[i], structure[i+1]))