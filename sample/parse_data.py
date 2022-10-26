import os
from scipy.io import loadmat
from PIL import Image
import numpy as np

# https://www.nist.gov/itl/products-and-services/emnist-dataset

__data_dir = os.path.join(os.path.dirname(__file__), "..", "data")


if not os.path.exists(os.path.join(__data_dir, "EMNIST")):
    raise Exception("Download the EMNIST dataset from Google Drive")

emnist_data = loadmat(os.path.join(__data_dir, "EMNIST", "emnist-letters.mat"))["dataset"][0][0][0][0][0]

def get_data(data):
    key = "abcdefghijklmnopqrstuvwxyz"
    for i, img in enumerate(data[0]):
        img_arr = np.flip(np.rot90(img.reshape((28, 28)), -1), 1)
        yield key[data[1][i][0]-1], img_arr

i = 0
for label, image in get_data(emnist_data):
    if i > 10: break
    Image.fromarray(image).save(f"image_tests/{label}.png")
    i += 1
