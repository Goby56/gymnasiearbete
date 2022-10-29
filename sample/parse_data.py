import os
from scipy.io import loadmat
from PIL import Image
import numpy as np

# https://www.nist.gov/itl/products-and-services/emnist-dataset


class CompiledDataset:
    """
    # CompiledDataset
    Used to represent the EMNIST datasets located at data\\EMNIST. Provides an easy way to gather batches of
    samples with a validation partition that, if specified also will be provided as a means to measure the 
    performance of the model. The length of this partition is determined by the length of the test data also
    accessible with this class.

    ## Args:
    dataset_filename: String argument needs to be a valid file name within the EMNIST data folder

    validation_partition: Booleon determining wheter or not to extract validation data
    """
    __data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    def __init__(self, dataset_filename: str, validation_partition = False):
        filepath = os.path.join(self.__data_dir, "EMNIST", dataset_filename)

        if not os.path.exists(filepath):
            raise Exception("Dataset not found! Download the EMNIST dataset from Google Drive")

        self._data = loadmat(filepath, simplify_cells = True)["dataset"]

        training_len = len(self._data["train"]["labels"])
        partition_len = len(self._data["test"]["labels"]) if validation_partition else 0
        
        self.training_data = self._create_labeled_generator("train", slice(0, training_len-partition_len))
        self.test_data = self._create_labeled_generator("test", slice(0, partition_len))
        self.validation_data = self._create_labeled_generator("train", slice(training_len-partition_len, training_len))

    def _create_labeled_generator(self, target: str, interval: slice):
        assert target in ["train", "test"], "arg. target not of type 'train', 'test' or 'validation'"
        targeted_data = self._data[target]
        for image, label in zip(targeted_data["images"][interval], targeted_data["labels"][interval]):
            yield (np.reshape(image, (28, 28)), label)

    def next_batch(self):
        # Yield a batch of size n
        pass


dataset = CompiledDataset("emnist-balanced.mat")

full_image = np.zeros((1, 28*4))
i = 0
for image, label in dataset.training_data:
    if i > 100: break
    row = image
    row = np.concatenate((image, np.rot90(image)), axis=1)
    row = np.concatenate((row, np.rot90(image, 2)), axis=1)
    row = np.concatenate((row, np.rot90(image, 3)), axis=1)

    full_image = np.concatenate((full_image, row), axis=0)
    i += 1

Image.fromarray(full_image).show()
