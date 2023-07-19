import numpy as np
import cv2
from tensorflow import keras
import modules.util as util

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X = self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        x1 = np.empty((self.batch_size, 224,224,3))
        x2 = np.empty((self.batch_size, 224,224,3))
        # x1 = np.empty((self.batch_size, 3, IMAGE_DIMS,IMAGE_DIMS))
        # x2 = np.empty((self.batch_size, 3, IMAGE_DIMS,IMAGE_DIMS))
        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # print(self.labels[int(ID)])
            path1, path2, label, s1, s2, sw= self.labels[int(ID)]

            _x1 = cv2.imread(path1)
            _x2 = cv2.imread(path2)
            _x1 = util.preprocess(_x1, s1)
            _x1 = util.scale_back(_x1) / 255.
            _x2 = util.preprocess(_x2, s2)
            _x2 = util.scale_back(_x2) / 255.

            x1[i,] = _x1
            x2[i,] = _x2
            y[i,] = label
        return [x1, x2], y, sw
