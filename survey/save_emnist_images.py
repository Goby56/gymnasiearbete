import sys, os
sys.path.append(os.getcwd())
import sample
from PIL import Image

def save_emnist_images(amount: int):
    """
    Only used to load and- save emnist images to be used in the survey.
    """
    dataset = sample.CompiledDataset(filename="emnist-balanced.mat")
    samples = dataset.get(amount, convert=True)

    occurences = {}
    for i in range(amount):
        label = samples[i][1]
        print(label)
        if label in occurences:
            occurences[label] += 1
        else:
            occurences[label] = 0            
        # pil_img = Image.fromarray(samples[i][0]).convert("RGB")
        fn = f"emnist_{label}({occurences[label]}).png"
        print(fn)
        
save_emnist_images(1000)

# def save_emnist_images(amount: int):
#     """
#     Only used to load and save emnist images to be used in the survey.
#     """
#     dataset = sample.CompiledDataset(filename="emnist-balanced.mat", image_size=(28, 28))
#     samples = dataset.get(amount, convert=True)
#     occurences = {}
#     for i in range(amount):
#         label = samples[i][1]
#         if label in occurences:
#             occurences[label] += 1
#         else:
#             occurences[label] = 0            
#         pil_img = Image.fromarray(samples[i][0]).convert("RGB")
#         fn = f"emnist_{samples[i][1]}({occurences[label]}).png"
#         path = SURVEY_IMAGES_PATH+f"\\{fn}"
#         pil_img.save(path)
