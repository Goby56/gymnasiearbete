import os
from scipy.io import loadmat
from PIL import Image
import numpy as np

# https://www.nist.gov/itl/products-and-services/emnist-dataset
__data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

emnist_data = loadmat(os.path.join(__data_dir, "EMNIST", "emnist-letters.mat"))

arr = emnist_data["dataset"][0][0][0][0][0][0][327]

arr = np.flip(np.rot90(arr.reshape((28, 28)), -1), 1)


Image.fromarray(arr).show()  # ave("letter.png")
