import numpy as np
from PIL import Image, ImageFilter
import random

def load_img():
    return Image.open("test.png")

def to_array(image):
    return np.asarray(image.convert("L")) / 255

def to_image(array):
    return Image.fromarray(array * 255)

def augment_data(image, options):
    new_img = image
    if options["rotate"]:
        new_img = new_img.rotate(random.randint(-30, 30), resample=Image.BICUBIC)
    if options["blur"]:
        new_img = new_img.filter(ImageFilter.GaussianBlur(random.random() * 1.2))
    if options["noise"]:
        array = to_array(new_img)
        array += np.random.random(array.shape) * 0.5

        return np.clip(array, 0, 1)

    return to_array(new_img)

options = {
    "noise": True,
    "shift": False, # no
    "rotate": True,
    "scale": False, # no
    "blur": True
        }

img = load_img()
arr = augment_data(img, options)
new_img = to_image(arr)
new_img.show()