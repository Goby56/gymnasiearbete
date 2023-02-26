import os, sys
from PIL import Image

sys.path.append(os.getcwd())

from sample import CompiledDataset

dataset = CompiledDataset(
    filename="emnist-letters.mat", 
    image_size=(28, 28)
)

for _ in dataset.next_batch(69):
    pass

for image, label in dataset.get(25, convert=True):
    Image.fromarray(image).convert("L").save(f"survey/images/emnist-{label}.png")