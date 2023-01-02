import os
from scipy.io import loadmat
import numpy as np
from typing import Union
from PIL import Image

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

    def __init__(self, *,
        filename: str,
        validation_partition = True, 
        standardize = False
    ):
        filepath = os.path.join(self.__data_dir, "EMNIST", filename)

        if not os.path.exists(filepath):
            raise Exception("Dataset not found! Download the EMNIST dataset from Google Drive")

        self.__data = loadmat(filepath, simplify_cells = True)["dataset"]
        self.is_standardized = standardize
        
        mapping = self.__data["mapping"][:, 1].flatten()
        self.labels = np.vectorize(chr)(mapping)
        

        img = self.__data["train"]["images"][0]
        lbl = self.__data["train"]["labels"][0]
        self.in_out_dim = (img.size, lbl.size)
        self.data_dim = (28, 28)

        self.training_len = len(self.__data["train"]["labels"])
        self.partition_len = len(self.__data["test"]["labels"])
        self.use_validation_partition = validation_partition
        
        self.generate_training_data()

    def convert_image(self, flat_array: np.ndarray) -> np.ndarray:
        new_array = flat_array.reshape(self.data_dim)
        if self.is_standardized:
            # Remap values to 0 - 255
            new_array = np.interp(new_array, (new_array.min(), new_array.max()), (0, 255)) 
        return new_array

    def convert_label(self, hot_vector: np.ndarray) -> str:
        return self.labels[np.where(hot_vector==1)[0]][0]

    def generate_training_data(self):
        training_interval = slice(0, self.training_len - self.partition_len * self.use_validation_partition)
        validation_interval = slice(self.training_len - self.partition_len * self.use_validation_partition, self.training_len)
        self.training_data = self.__sample_generator("train", training_interval)
        self.validation_data = self.__sample_generator("train", validation_interval)

    def get_test_data(self):
        return self.__sample_generator("test", slice(0, self.partition_len))

    def get(self, amount: int, convert=False):

        def conv(data):
            images, labels = zip(*data)
            images = map(self.convert_image, images)
            labels = map(self.convert_label, labels)
            return list(zip(images, labels))

        data = self.next_batch(amount)
        if amount > 1:
            return conv(data) if convert else data
        return conv(data)[0] if convert else data

    def next_batch(self, batch_size: int):
        for _ in range(batch_size):
            sample = next(self.training_data, None)
            if sample is None:
                break
            yield sample

    def __to_hot_vector(self, index_int: int) -> Union[str, np.ndarray]:
        out = np.zeros(len(self.labels))
        out[index_int-1] = 1
        return out

    def __sample_generator(self, target: str, interval: slice):
        assert target in ["train", "test"], "arg. target not of type 'train' or 'test'"
        targeted_data = self.__data[target]
        for image, label in zip(targeted_data["images"][interval], targeted_data["labels"][interval]):
            image = np.flip(np.rot90(np.reshape(image, self.data_dim), -1), -1).flatten()

            if self.is_standardized:
                image = self.__standardized_image(image)
            
            label = self.__to_hot_vector(label)

            yield (image, label)

    def __standardized_image(self, image):
        return (image - np.mean(image)) / np.std(image)
     

if __name__ == "__main__":
    dataset = CompiledDataset(filename="emnist-letters.mat", validation_partition=True)
    for data in dataset.next_batch(1):
        print(data)