import numpy as np

import os, sys
sys.path.append(os.getcwd())
import sample

model = sample.Model("test_model")
network = sample.Network(model)
dataset = sample.CompiledDataset(
    filename=model.dataset, 
    validation_partition=True, 
    as_array=True,
    flatten=True,
    normalize=True
)

for _ in dataset.next_batch(1234):
    pass

image, correct_guess = next(dataset.next_batch(1))
guess = network.forward(image.reshape(1, 784)).flatten()
hot_guess = (guess == np.max(guess)).astype(int)

image, labeled_guess = dataset.represent(image, hot_guess)
dataset.show(image, labeled_guess)
print(guess)



