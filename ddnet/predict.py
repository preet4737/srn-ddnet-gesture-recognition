import os
import logging
import numpy as np
import keras
from .labels import _LABELS_14, _LABELS_28


class Predictor:
    def __init__(self):
        global _LABELS_28, _LABELS_14
        self.labels = _LABELS_14

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model = keras.models.load_model(dir_path + "/model")

    def __call__(self, x: list):
        x = np.stack(x)
        x = np.reshape(x, (1, 30, 21, 3))
        y = self.model.predict(x)
        y = np.argmax(y)
        return self.labels[y]

    def predict(self, x: list):
        return self.__call__(x)

