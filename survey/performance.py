import os, sys, re, json
sys.path.append(os.getcwd())
import sample
from PIL import Image, ImageDraw, ImageFont
from typing import NamedTuple

labels = sample.CompiledDataset(filename="emnist-balanced.mat").labels

name = input()
GUESSES_PATH = os.path.join(os.getcwd(), f"survey\\guesses\\{name}")
IMAGE_PATH = os.path.join(os.getcwd(), f"survey\\images")

class Guess(NamedTuple):
    ans: str
    guess: str
    fname: str

with open(GUESSES_PATH, "r") as f:
    guesses_dict = json.load(f)
    guesses = []
    for k, v in guesses_dict.items():
        ans = re.search("(?<=_)\S", k).group()
        guesses.append(Guess(ans, v, k))

def is_wrong_guess(correct: str, guess: str):
    if correct == guess:
        return False
    if guess.isalpha() and guess not in labels:
        return False
    return True

wrong_guesses = [g for g in guesses if is_wrong_guess(g.ans, g.guess)]

accuracy = 1-len(wrong_guesses)/len(guesses)
print(accuracy)

font_size = 12
font = ImageFont.truetype("arial.ttf", font_size)
upscaling = 1
occurences = {}
wrong_sheet = Image.new("RGB", (upscaling*28*len(wrong_guesses), upscaling*28))
for i, g in enumerate(wrong_guesses):
    if g.ans not in occurences:
        occurences[g.ans] = 0
    occurences[g.ans] += 1

    img = Image.open(IMAGE_PATH+f"\\{g.fname}")
    img = img.resize((upscaling*28, upscaling*28))
    draw = ImageDraw.Draw(img)
    text = Image.new("RGB", (upscaling*28, font_size))
    draw.text((0,0), g.guess, "red", font)
    draw.text((0,upscaling*14), g.ans, "green", font)
    
    wrong_sheet.paste(img, (upscaling*28*i, 0))


occurences = {v: k for k, v in occurences.items()}

top_five_wrong = [occurences[k] for k in sorted(occurences.keys(), reverse=True)]


wrong_sheet.show()
print(top_five_wrong)
