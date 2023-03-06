import os, sys, re
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())

import sample

PATH_EXTRA_IMAGES = os.path.join(os.getcwd(), "survey\\extra_images")
PATH_IMAGES = os.path.join(os.getcwd(), "survey\\images")

model = sample.Model(input("AI name: "))
network = sample.Network(model)

res_cache = [0, 0] # num failed, num sucsess

def guess(image, ans):
    array = np.asarray(image)
    array.shape = (1, array.size)

    out_vec = network.forward(array)
    i = np.where(out_vec == out_vec.max())[1][0]
    res_cache[ans == model.mapping[i]] += 1

def get_ans(filename):
    return re.search("(?<=_).", filename)[0]

images = [(os.path.join(PATH_IMAGES, filename), get_ans(filename)) 
          for filename in os.listdir(PATH_IMAGES)]
extra_images = [(os.path.join(PATH_EXTRA_IMAGES, filename), 
                 get_ans(filename)) for filename in os.listdir(PATH_EXTRA_IMAGES)]

for path, key in images + extra_images:
    image = Image.open(path).convert("L")
    guess(image, key)

print(res_cache)



    