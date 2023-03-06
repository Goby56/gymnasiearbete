import os, re, json
from PIL import Image
import collections

name = input()
GUESSES_PATH = os.path.join(os.getcwd(), f"survey\\guesses\\{name}")
IMAGE_PATH = os.path.join(os.getcwd(), f"survey\\images")

with open(GUESSES_PATH, "r") as f:
    guesses = json.load(f)

guess = collections.namedtuple("guess", ["ans", "guess", "filename"])
wrong_guess = []

for k, v in guesses.items():
    ans = re.sub("(?<=_)\S", "", k)
    if ans != v:
        wrong_guess.append(guess(ans=ans, guess=v, filename=k))

