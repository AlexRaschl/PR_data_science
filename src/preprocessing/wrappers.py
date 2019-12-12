import numpy as np


class FramePredictions(object):
    def __init__(self, labels: np.ndarray, probabilities: np.ndarray):
        self._labels = labels.copy()
        self._probabilities = probabilities.copy()

    def __str__(self):
        return np.array_str(self.labels) + np.array_str(self.probabilities)

    @property
    def labels(self):
        return self._labels

    @property
    def probabilities(self):
        return self._probabilities

    def return_label_descr(self):
        raise NotImplementedError('Not implemented yet!')
