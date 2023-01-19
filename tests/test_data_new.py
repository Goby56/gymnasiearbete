import os, sys
sys.path.append(os.getcwd())

import numpy as np
from PIL import Image

import sample.data as new
import sample.data_old as old

FILE = "emnist-letters.mat"

new_data = new.CompiledDataset(
    filename=FILE,
    image_size=(28, 28),
    standardize=True
)

old_data = old.CompiledDataset(
    filename=FILE,
    validation_partition=True, 
    as_array=True, 
    flatten=True,
    normalize=True
)

dn, ln = next(new_data.next_batch(1))
do, lo = next(old_data.next_batch(1))

assert dn.all() == do.all()
assert ln.all() == lo.all()

n = new_data.convert_image(dn)
o = new_data.convert_image(do)

n = new_data.convert_label(ln)
o = new_data.convert_label(lo)
print(n, o)



