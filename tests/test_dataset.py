import sample
from PIL import Image
import numpy as np


dataset = sample.CompiledDataset(
    filename="emnist-letters.mat"
)

test_data = dataset.get_test_data()

image1, label1 = dataset.get(1, convert=True)
pil_image1 = Image.fromarray(image1)

image2, label2 = next(test_data)
image2 = dataset.convert_image(image2)
label2 = dataset.convert_label(label2)
pil_image2 = Image.fromarray(image2)

pil_image1.show()
print(label1)

pil_image2.show()
print(label2)



