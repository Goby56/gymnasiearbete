import numpy as np

def str_reader(string: str):
    weights = []
    biases = []
    for layer in string.split("\n\n")[:-1]:
        lines = layer.split("\n")
        weights.append(np.array([np.array(np.array(list(map(float, w)))) for w in [weight.split(";")[:-1] for weight in lines[:-1]]]))
        biases.append(np.array([float(bias) for bias in lines[-1].split(";")[:-1]]))
    return (np.array(weights, dtype=object),
            np.array(biases, dtype=object))

def file_reader(file):
    return str_reader(file.read())

def str_writer(weights, biases) -> str:
    string = ""
    join = lambda s: ";".join([str(x) for x in s]) + ";"
    for weight, bias in zip(weights, biases):
        for w in weight:
            string += f"{join(w)}\n"
        string += f"{join(bias)}\n\n"
    return string

if __name__ == "__main__":
    import os
    path = os.path.join(os.path.dirname( __file__ ), "..", "data\\models\\example_file.wnb")
    with open(path, "r") as file:
        w, b = file_reader(file)
        #print(w[0])
    # with open(path, "w") as file:
    #     file.write(str_writer(w, b))