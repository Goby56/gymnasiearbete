import os
from scipy.io import loadmat

# https://www.nist.gov/itl/products-and-services/emnist-dataset
__data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))

loadmat(os.path.join(__data_dir, "EMNIST"))

