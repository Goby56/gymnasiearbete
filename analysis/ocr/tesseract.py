from PIL import Image
from pytesseract import pytesseract
import os

path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path_to_images = r'survey/images/'

pytesseract.tesseract_cmd = path_to_tesseract

guesses = 0
correct_guesses = 0

for root, dirs, file_names in os.walk(path_to_images):
    for file_name in file_names:    
        img = Image.open(path_to_images + file_name)
        guess = pytesseract.image_to_string(
            img, config=("-c tessedit"
                        "_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                        " --psm 10"
                        " -l osd"
                        " ")).strip()
        
        index = file_name.index("(")
        answer = file_name[index-1].lower()

        guesses += 1
        if guess == answer:
            correct_guesses += 1
        
        
print(correct_guesses/guesses)