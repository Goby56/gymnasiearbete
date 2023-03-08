import os, sys
from PIL import Image

sys.path.append(os.getcwd())
from sample import CompiledDataset

IMAGES_FOLDER = os.path.join(os.getcwd(), "analysis\\balanced_images")

dataset = CompiledDataset(
    filename="emnist-balanced.mat" 
)

for i, sample in enumerate(dataset.get(150, convert=True, target="test")):
    Image.fromarray(sample[0]).convert("RGB").save(
        os.path.join(IMAGES_FOLDER, f"emnist_{sample[1]}({i}).png")
    )

