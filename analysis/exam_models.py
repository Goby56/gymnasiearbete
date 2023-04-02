import os, sys, re, itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.getcwd())
upscaling = 2
font = ImageFont.truetype("arial.ttf", 12*upscaling)

import sample

PATH_EXTRA_IMAGES = os.path.join(os.getcwd(), "survey\\extra_images")
PATH_SURVEY_IMAGES = os.path.join(os.getcwd(), "survey\\images")
PATH_BALANCED_IMAGES = os.path.join(os.getcwd(), "analysis\\balanced_images")
PATH_DIGIT_IMAGES = os.path.join(os.getcwd(), "analysis\\digit_images")
PATH_RESULT = os.path.join(os.getcwd(), "survey\\results")

print("Image sets available: 300 | 200 | 100")
# 300 is used for model choosing, 200 is for comparison (AI vs Human)
# 100 is digits used for specialized vs general 

def corrected_guess_overlay(img28x28: Image.Image, correct: str, guess: str, is_emnist: bool):
    img = img28x28.convert("RGB").resize((upscaling*28, upscaling*28), resample=Image.Resampling.BOX)
    draw = ImageDraw.Draw(img)
    draw.text((upscaling*1,0), guess, "red", font)
    draw.text((upscaling*1,upscaling*14), correct, "green", font)
    if is_emnist:
        draw.rectangle((0,0,upscaling*28-1,upscaling*28-1), None, "blue", upscaling*1)
    return np.asarray(img)

while True:
    model_name, image_set = input("Model name & image set -> ").split(" ")
    if image_set not in ["300", "200", "100"]:
        print(f"Not an available image set: {image_set}")
        continue
    model = sample.Model(model_name)
    network = sample.Network(model)

    res_cache = [0, 0] # num failed, num sucsess
    wrong_guesses = []

    def guess(image, key, path):
        array = np.asarray(image)
        array.shape = (1, array.size)

        out_vec = network.forward(array)
        i = np.where(out_vec == out_vec.max())[1][0]
        ans = model.mapping[i]
        correct = key == ans or (ans.isalpha() and not key in model.mapping 
                                   and ans.upper() == key.upper())
        res_cache[correct] += 1
        if not correct:
            wrong_guesses.append((image, key, ans, "emnist" in path))

    def get_ans(filename):
        return re.search("(?<=_).", filename)[0]
    
    if image_set == "300":
        images = [(os.path.join(PATH_SURVEY_IMAGES, filename), get_ans(filename)) 
                for filename in os.listdir(PATH_SURVEY_IMAGES) if not "emnist" in filename]
        images += [(os.path.join(PATH_EXTRA_IMAGES, filename), get_ans(filename)) 
                   for filename in os.listdir(PATH_EXTRA_IMAGES)]
        images += [(os.path.join(PATH_BALANCED_IMAGES, filename), get_ans(filename)) 
                   for filename in os.listdir(PATH_BALANCED_IMAGES)]
    elif image_set == "200":
        images = [(os.path.join(PATH_SURVEY_IMAGES, filename), get_ans(filename)) 
                for filename in os.listdir(PATH_SURVEY_IMAGES)]
    elif image_set == "100":
        images = [(os.path.join(PATH_DIGIT_IMAGES, filename), get_ans(filename)) 
                  for filename in os.listdir(PATH_DIGIT_IMAGES)]
        
    for path, key in images:
        image = Image.open(path).convert("L")
        guess(image, key, path.split("\\")[-1])

    arrays = itertools.starmap(corrected_guess_overlay, wrong_guesses)
    Image.fromarray(np.concatenate(list(arrays), axis=1)).save(PATH_RESULT + f"\\{model_name}-{image_set}(with borders).png")
    print(res_cache, res_cache[1]/sum(res_cache))