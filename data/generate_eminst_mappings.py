import os, sys
sys.path.append(os.getcwd())

import numpy as np
import sample

PATH = os.path.join(os.getcwd(), "data\\EMNIST")
assert os.path.exists(PATH)

for model_name in os.listdir(PATH):
    labels = sample.CompiledDataset(filename=model_name, image_size=(28, 28)).labels
    np.save(os.path.join(PATH, f"{model_name}.npy"), labels)