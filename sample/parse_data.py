import os
from scipy.io import loadmat
from PIL import Image
import numpy as np


class CompiledDataset:
    """
    ## CompiledDataset
    Used to represent the EMNIST datasets located at "data\\EMNIST". Provides an easy way to gather batches of
    samples which has a validation partition that, if specified, will be provided as a means to measure the 
    performance of the model. The length of this partition is determined by the length of the test data which
    also is accessible with this class.

    ### Args:
    dataset_filename: String argument needs to be a valid file name within the EMNIST data folder

    validation_partition: Booleon determining wheter or not to extract validation data

    https://www.nist.gov/itl/products-and-services/emnist-dataset
    """
    __data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    def __init__(self, dataset_filename: str, validation_partition = False):
        filepath = os.path.join(self.__data_dir, "EMNIST", dataset_filename)

        if not os.path.exists(filepath):
            raise Exception("Dataset not found! Download the EMNIST dataset from Google Drive")

        self._data = loadmat(filepath, simplify_cells = True)["dataset"]

        training_len = len(self._data["train"]["labels"])
        partition_len = len(self._data["test"]["labels"])
        
        self.training_data = self.__create_labeled_generator("train", slice(0, training_len - partition_len*validation_partition))
        self.test_data = self.__create_labeled_generator("test", slice(0, partition_len))
        self.validation_data = self.__create_labeled_generator("train", slice(training_len - partition_len*validation_partition, training_len))

    def __create_labeled_generator(self, target: str, interval: slice):
        assert target in ["train", "test"], "arg. target not of type 'train', 'test' or 'validation'"
        targeted_data = self._data[target]
        for image, label in zip(targeted_data["images"][interval], targeted_data["labels"][interval]):
            char = chr(self._data["mapping"][label][1])
            arr = np.flip(np.rot90(np.reshape(image, (28, 28)), -1), -1)
            yield (arr, char)

    def next_batch(self, batch_size: int):
        for _ in range(batch_size):
            yield next(self.training_data)


dataset = CompiledDataset("emnist-balanced.mat", validation_partition=True)
