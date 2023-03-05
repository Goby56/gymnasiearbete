import os, re, json
from PIL import Image
name = input()
GUESSES_PATH = os.path.join(os.getcwd(), f"survey\\guesses\\{name}")
IMAGE_PATH = os.path.join(os.getcwd(), f"survey\\images")

with open(GUESSES_PATH, "r") as f:
    guesses = json.load(f)

incorrect_img = []
correct = 0
for k, v in guesses.items():
    letter = re.sub("\(\d\)", "", k)
    if re.search(".(?=\.png)", letter).group() == v:
        correct += 1
    else:
        incorrect_img.append((k, v))

print(correct / len(guesses))

for img_name, guess in incorrect_img:
    if "emnist" in img_name:
        continue
    print(guess, img_name)
    if input("Show?") == "y":
        img = Image.open(IMAGE_PATH+f"\\{img_name}")
        img.show()
        input()
