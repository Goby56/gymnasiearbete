import os
from scipy.io import loadmat
from PIL import Image
import numpy as np

# https://www.nist.gov/itl/products-and-services/emnist-dataset


class CompiledDataset:
    __data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    def __init__(self, dataset_filename: str, validation_partition = False):
        filepath = os.path.join(self.__data_dir, "EMNIST", dataset_filename)

        if not os.path.exists(filepath):
            raise Exception("Dataset not found! Download the EMNIST dataset from Google Drive")

        self._data = loadmat(filepath, simplify_cells = True)["dataset"]

        training_len = len(self._data["train"]["labels"])
        partition_len = len(self._data["test"]["labels"])
        
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
