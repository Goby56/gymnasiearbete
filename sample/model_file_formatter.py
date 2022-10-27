import numpy as np

def str_formatter(string: str):
    network = {
        "weights": [],
        "biases": []
    }
    weights = []
    biases = []
    for line in string.split("\n"):
        if line == "":
            network["weights"].append(weights)
            network["biases"].append(biases)
            weights = []
            biases = []
            continue
        elements = line.split(";")[:-1]
        weights.append([int(w) for w in elements[:-1]])
        biases.append(elements[-1].split("(b)")[1])
    return (np.asarray(network["weights"], dtype=object), 
            np.asarray(network["biases"], dtype=object))


def file_formatter(file):
    return str_formatter(file.read())