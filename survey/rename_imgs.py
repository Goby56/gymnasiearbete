import os, re

path = os.path.join(os.getcwd(), "survey\\images")
names = [n for n in os.listdir(path) if re.search("\w+_(lo|up|nu)_.\.png", n)]
prev = ""
for i in range(len(names)):
    name = re.sub("_(lo|up|nu)(?=_)", "", names[i])
    name = name.replace(".png", "")
    letter = name.split("_")[1]
    if letter in prev:
        names[i] = f"{name}(1).png"
        continue
    prev += letter
    names[i] = f"{name}(0).png"

print(names)