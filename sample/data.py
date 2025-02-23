import os
from scipy.io import loadmat
import numpy as np
from typing import Union
from PIL import Image
import random

IMAGE_SIZE = (28, 28)

def convert_label(hot_vector: np.ndarray, mapping: list) -> str:
    """
    Arguments:
        hot_vector: np.ndarray --- a one-hot vector of type np.ndarray
    Returns:
        str --- the character that the one-hot vector represents acording to dataset mapping
    """
    labels = np.asarray(mapping)
    return labels[np.where(hot_vector==1)[0]][0]


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
        data_augmentation: dict = {},
        subtract_label = False
    ):
        filepath = os.path.join(self.__data_dir, "EMNIST", filename)

        if not os.path.exists(filepath):
            raise Exception("Dataset not found! Download the EMNIST dataset from Google Drive")

        self.__data = loadmat(filepath, simplify_cells = True)["dataset"]
        self.__subtract_label = subtract_label

        self.validation_partition = validation_partition

        self.augmentations = {
            "rotate": False, 
            "noise": False,
            "shift": False
        } | data_augmentation

        mapping = self.__data["mapping"][:, 1].flatten()
        self.labels = np.vectorize(chr)(mapping)

        self.__load_generators()
        self.shape = tuple(map(len, next(self.next_batch(1))))

    @classmethod
    def get_mapping(cls, dataset: str):
        filepath = os.path.join(cls.__data_dir, "EMNIST", dataset)
        data = loadmat(filepath, simplify_cells = True)["dataset"]
        mapping = data["mapping"][:, 1].flatten()
        return np.vectorize(chr)(mapping)

    #region private shit
    def __load_generators(self):
        training_len = len(self.__data["train"]["labels"])
        partition_len = len(self.__data["test"]["labels"])

        self.__new_training_data = lambda: self.__sample_gen(
            "train", slice(0, training_len - partition_len*self.validation_partition)
        )
        self.__new_validation_data = lambda: self.__sample_gen(
            "train", slice(training_len - partition_len*self.validation_partition, training_len)
        )
        self.__new_test_data = lambda: self.__sample_gen("test", slice(0, partition_len))

        self.training_data = self.__new_training_data()
        self.test_data = self.__new_test_data()
        self.validation_data = self.__new_validation_data()

        self.training_len = training_len - partition_len*self.validation_partition
        self.validation_len = partition_len*self.validation_partition

    def __sample_gen(self, target: str, interval: slice):
        assert target in ["train", "test"], "arg. target not of type 'train', 'test' or 'validation'"
        targeted_data = self.__data[target]
        for image, label in zip(targeted_data["images"][interval], targeted_data["labels"][interval]):
            image = np.flip(np.rot90(np.reshape(image, IMAGE_SIZE), -1), -1)

            if target == "train":
                if self.augmentations["rotate"]:
                    new_img = Image.fromarray(image, "L")
                    new_img = new_img.rotate(random.randint(-30, 30), resample=Image.BICUBIC)
                    image = np.asarray(new_img)

                if self.augmentations["shift"]:
                    array = np.asarray(image)
                    for i in range(IMAGE_SIZE[0]):
                        if np.any(array[:,i]) or np.any(array[:,IMAGE_SIZE[0]-1-i]):
                            break
                    image = np.roll(array, random.randint(-i, i))

                if self.augmentations["noise"]:
                    new_img = (image + np.random.random(image.shape) * 255 * 0.5).astype(int)
                    image = np.clip(new_img, 0, 255)
            
            image=image.flatten()     
            label = self.__to_hot_vector(label)

            yield (image, label)

    def __to_hot_vector(self, index_int: int) -> Union[str, np.ndarray]:
        out = np.zeros(len(self.labels))
        out[index_int-self.__subtract_label] = 1
        return out

    #endregion private shit

    def next_batch(self, batch_size: int):
        for _ in range(batch_size):
            sample = next(self.training_data, None)
            if sample is None:
                self.training_data = self.__new_training_data()
                next(self.training_data)
                break
            yield sample

    def next_test_batch(self, batch_size: int):
        for _ in range(batch_size):
            sample = next(self.test_data, None)
            if sample is None:
                self.test_data = self.__new_test_data()
                next(self.test_data)
                break
            yield sample

    def convert_label(self, hot_vector: np.ndarray) -> str:
        """
        Arguments:
            hot_vector: np.ndarray --- a one-hot vector of type np.ndarray
        Returns:
            str --- the character that the one-hot vector represents acording to dataset mapping
        """
        return convert_label(hot_vector, self.labels)

    def get(self, amount: int, convert: bool = False, target: str = "train") -> tuple[np.ndarray, Union[str, np.ndarray]]:
        """
        OBS! This function exasuts the training data generator
        Arguments:
            amount: int --- the amount of data-points to be retrived
            convert: bool = False --- if the data-point(s) should be converted into thier readable formats
            target: str = 'train' --- target dataset, can be 'train' or 'test'
        Returns:
            tuple[np.ndarray, Union[str, np.ndarray]] --- returns a tuple with the image and the label, 
                                                          if converted the label is a string else a np.ndarray
        """
        assert target in ["train", "test"], f"arg. target must be 'train' or 'test' not {target}"
        
        def conv(data):
            images, labels = zip(*data)
            images = map(lambda arr: arr.reshape(28,28), images)
            labels = map(self.convert_label, labels)
            return list(zip(images, labels))

        if target == "train":
            data = self.next_batch(amount)
        else:
            data = self.next_test_batch(amount)
        if amount > 1:
            return conv(data) if convert else next(data)
        return conv(data)[0] if convert else next(data)
