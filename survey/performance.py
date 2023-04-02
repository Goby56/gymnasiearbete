import os, sys, re, json
sys.path.append(os.getcwd())
import sample
from PIL import Image, ImageDraw, ImageFont
from typing import NamedTuple

labels = sample.CompiledDataset(filename="emnist-balanced.mat").labels
upscaling = 2
font_size = upscaling*12
font = ImageFont.truetype("arial.ttf", font_size)
label_font = ImageFont.truetype("arial.ttf", int(font_size*1.75))
IMAGE_PATH = os.path.join(os.getcwd(), f"survey\\images")
WRONG_SHEET_PATH = os.path.join(os.getcwd(), f"survey\\results")
GUESSES_PATH = os.path.join(os.getcwd(), f"survey\\guesses")
ALFA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class Guess(NamedTuple):
    ans: str
    guess: str
    fname: str

def is_wrong_guess(correct: str, guess: str):
    if correct == guess:
        return False
    if guess.isalpha() and guess not in labels:
        return False
    return True

def get_performance(json_path):
    with open(json_path, "r") as f:
        guesses_dict = json.load(f)
        guesses = []
        for k, v in guesses_dict.items():
            ans = re.search("(?<=_)\S", k).group()
            guesses.append(Guess(ans, v, k))
    wrong_guesses = [g for g in guesses if is_wrong_guess(g.ans, g.guess)]
    accuracy = 1-len(wrong_guesses)/len(guesses)
    return accuracy, wrong_guesses


if __name__ == "__main__":
    available_names = os.listdir(GUESSES_PATH)
    if name := input("Get performance of: " + ", ".join(available_names) + ".\nLeave empty for all of the above. -> "):
        if name not in available_names: quit()
        available_names = [name]

    occurences = {}
    performances = []
    for name in available_names:
        path = os.path.join(os.getcwd(), f"survey\\guesses\\{name}")
        accuracy, wrong_guesses = get_performance(path)
        performances.append((accuracy, wrong_guesses))
        for guess in wrong_guesses:
            if guess.fname not in occurences:
                occurences[guess.fname] = 0
            occurences[guess.fname] += 1

    performances.sort(key=lambda e: e[0])

    ours = list(map(lambda x: x[0], sorted([(fname, n) for fname, n in occurences.items() if "emnist" not in fname], 
                  key=lambda x: x[1], reverse=True)))
    emnist = list(map(lambda x: x[0], sorted([(fname, n) for fname, n in occurences.items() if "emnist" in fname], 
                    key=lambda x: x[1], reverse=True)))


    X = (upscaling*28*(len(available_names)*2+1))
    Y = upscaling*28*(len(emnist)+1)

    wrong_sheet = Image.new("RGB", (X, Y))

    for c in range(len(available_names)):
        accuracy, wrong_guesses = performances[c]

        sheet_draw = ImageDraw.Draw(wrong_sheet)
        sheet_draw.text((c*upscaling*28+upscaling*28//5, 0), ALFA[c], "white", label_font)
        sheet_draw.text(((c+len(available_names)+1)*upscaling*28+upscaling*28//5, 0), ALFA[c], "white", label_font)

        for guess in wrong_guesses:
            if "emnist" in guess.fname:
                offset = len(available_names) + 1
                r = emnist.index(guess.fname)
            else:
                offset = 0
                r = ours.index(guess.fname)

            img = Image.open(IMAGE_PATH+f"\\{guess.fname}")
            img = img.resize((upscaling*28, upscaling*28), resample=Image.Resampling.BOX)
            draw = ImageDraw.Draw(img)
            draw.text((0,0), guess.guess, "red", font)
            draw.text((0,upscaling*14), guess.ans, "green", font)
            
            x = upscaling*28*(c+offset)
            y = upscaling*28*(r+1)
            
            wrong_sheet.paste(img, (x, y))

    vert_separator = Image.new("RGB", (int(upscaling*28*0.25), Y), "white")
    horz_separator = Image.new("RGB", (X, int(upscaling*28*0.25)), "white")
    wrong_sheet.paste(vert_separator, (upscaling*28*len(available_names)+int(upscaling*28*0.375), 0))
    wrong_sheet.paste(horz_separator, (0, int(upscaling*28*0.75)))

    # labels = Image.new("RGB", (X, upscaling*28))
    # draw = ImageDraw.Draw(labels)
    # draw.text((0,0), "Framtagna", "white", header_font)
    # draw.text((X//2,0), "EMNIST", "white", header_font)

    # final_img = Image.new("RGB", (X, Y+upscaling*28))
    # final_img.paste(header_img, (0,0))
    # final_img.paste(wrong_sheet, (0, upscaling*28))


    wrong_sheet.show()
    wrong_sheet.save(WRONG_SHEET_PATH + "\\large_sheet.png")
    


# occurences = {v: k for k, v in occurences.items()}

# top_five_wrong = [occurences[k] for k in sorted(occurences.keys(), reverse=True)]


# wrong_sheet.show()
# # wrong_sheet.save(WRONG_SHEET_PATH + f"\\{name}.png")

# print(len(wrong_guesses))
