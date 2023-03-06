import sys, os, re
sys.path.append(os.getcwd())
import sample
from PIL import Image
import numpy as np

IMAGES_FOLDER = os.path.join(os.getcwd(), "survey\\images")

def save_emnist_images(amount: int):
    """
    Only used to load and- save emnist images to be used in the survey.
    """
    dataset = sample.CompiledDataset(filename="emnist-balanced.mat")
    for i, data in enumerate(dataset.get(amount, convert=True)):
        Image.fromarray(data[0]).convert("RGB").save(
            os.path.join(IMAGES_FOLDER, f"emnist_{data[1]}({i+100}).png")
        )

def rename_shit_names():
    folder = os.listdir(IMAGES_FOLDER)
    for i, filename in enumerate(folder):
        new_name = re.sub("_\S{2}(?=\_)", "", filename)[:-4]
        new_name = f"{new_name}({i}).png"
        os.rename(os.path.join(IMAGES_FOLDER, filename),
                  os.path.join(IMAGES_FOLDER, new_name))

def check_images(amount: int):
    dataset = sample.CompiledDataset(filename="emnist-balanced.mat")

    labels = np.empty((amount, amount), dtype=str)
    image = None
    for i in range(amount):
        row, labels[i, 0] = dataset.get(1, convert=True)
        for j, data in enumerate(dataset.get(amount-1, convert=True)):
            row = np.concatenate((row, data[0]), axis=1)
            labels[i, j+1] = data[1]
        image = row if image is None else np.concatenate((image, row), axis=0)
    
    Image.fromarray(image).show()
    print(labels)
