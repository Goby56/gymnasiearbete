import numpy as np

def str_reader(string: str):
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
        weights.append([float(w) for w in elements[:-1]])
        biases.append(float(elements[-1].split("(b)")[1]))
    return (np.asarray(network["weights"], dtype=object), 
            np.asarray(network["biases"], dtype=object))

def file_reader(file):
    return str_reader(file.read())

def str_writer(weights, biases) -> str:
    string = ""
    for weight, bias in zip(weights, biases):
        for w, b in zip(weight, bias):
            ws = ";".join([str(x) for x in w])
            string += f"{ws};(b){b};\n"
        string += "\n"
    return string



#liam berg var here