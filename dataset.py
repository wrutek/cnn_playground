from tensorflow.contrib.keras import datasets
import numpy as np


class MnistDataset:
    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.mnist.load_data()
        print('Dataset loaded')

        self.train_images = self.train_images/255.0  # type: np.ndarray
        self.test_images = self.test_images/255.0  # type: np.ndarray
        self.train_images = self.train_images.reshape((-1, 28, 28, 1))
        self.test_images = self.test_images.reshape((-1, 28, 28, 1))

        self.train_labels = self.train_labels.astype(np.int)  # type: np.ndarray
        self.test_labels = self.test_labels.astype(np.int)  # type: np.ndarray
