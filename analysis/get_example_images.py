import os, sys, random
from PIL import Image, ImageFont, ImageDraw
import numpy as np

sys.path.append(os.getcwd())
from sample import CompiledDataset

IMAGES_FOLDER = os.path.join(os.getcwd(), "analysis\\example_images")
upscaling = 2
font = ImageFont.truetype("arial.ttf", 12*upscaling)

dataset = CompiledDataset(
    filename="emnist-balanced.mat" 
)

def label_img(img28x28: Image.Image, label):
    img = img28x28.convert("RGB").resize((upscaling*28, upscaling*28), resample=Image.Resampling.BOX)
    draw = ImageDraw.Draw(img)
    draw.text((upscaling*1,upscaling*14), label, "green", font)
    return img

n = int(input("How many? -> "))

while True:
    samples = list(dataset.get(n+random.randint(0, 1000), convert=True, target="test"))[-5:]
    if input("Use these: " + ", ".join(map(lambda x: x[1], samples)) + "? (y/n)") == "y":
        base_img = Image.new("RGB", (upscaling*28, upscaling*28*n))

        for i, sample in enumerate(samples):
            img = Image.fromarray(sample[0]).convert("RGB")
            labeled_img = label_img(img, sample[1])
            base_img.paste(labeled_img, (0, upscaling*28*i))

        base_img.save(os.path.join(IMAGES_FOLDER, f"emnist_{sample[1]}({i}).png"))
        break