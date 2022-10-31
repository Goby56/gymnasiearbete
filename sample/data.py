import os
from scipy.io import loadmat
import numpy as np
from typing import Union

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

    def __init__(self, dataset_filename: str, validation_partition = False, 
                 as_array = False, flatten = False):
        filepath = os.path.join(self.__data_dir, "EMNIST", dataset_filename)

        if not os.path.exists(filepath):
            raise Exception("Dataset not found! Download the EMNIST dataset from Google Drive")

        self.__data = loadmat(filepath, simplify_cells = True)["dataset"]
        self.__as_array = as_array
        self.__flatten = flatten

        training_len = len(self.__data["train"]["labels"])
        partition_len = len(self.__data["test"]["labels"])
        
        self.training_data = self.__create_labeled_generator("train", slice(0, training_len - partition_len*validation_partition))
        self.test_data = self.__create_labeled_generator("test", slice(0, partition_len))
        self.validation_data = self.__create_labeled_generator("train", slice(training_len - partition_len*validation_partition, training_len))

    def __label_type(self, label: str) -> Union[str, np.ndarray]:
        index = label-self.__data["mapping"][0][0]
        if not self.__as_array: return chr(self.__data["mapping"][index][1])
        out = np.zeros(len(self.__data["mapping"]))
        out[index] = 1
        return out

    def __create_labeled_generator(self, target: str, interval: slice):
        assert target in ["train", "test"], "arg. target not of type 'train', 'test' or 'validation'"
        targeted_data = self.__data[target]
        for image, label in zip(targeted_data["images"][interval], targeted_data["labels"][interval]):
            image = np.flip(np.rot90(np.reshape(image, (28, 28)), -1), -1)
            yield (image.flatten() if self.__flatten else image,
                   self.__label_type(label))

    def next_batch(self, batch_size: int):
        for _ in range(batch_size):
            yield next(self.training_data)


if __name__ == "__main__":
    dataset = CompiledDataset("emnist-letters.mat", validation_partition=True, as_array=True)
    for data in dataset.next_batch(1):
        print(data)