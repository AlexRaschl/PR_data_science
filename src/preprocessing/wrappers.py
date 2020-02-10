import os
import pickle as pkl

import numpy as np

from src.config import LABEL_TO_STRING_PATH, CACHED_LABEL_STRINGS


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
        if not os.path.exists(CACHED_LABEL_STRINGS):
            with open(LABEL_TO_STRING_PATH, 'r') as f:
                label_map = eval(f.read())
                os.makedirs(os.path.dirname(CACHED_LABEL_STRINGS), exist_ok=True)
                pkl.dump(label_map, open(CACHED_LABEL_STRINGS, 'wb'))

        label_map = pkl.load(open(CACHED_LABEL_STRINGS, 'rb'))

        return list(map(lambda l: label_map.get(l), self._labels))

    @staticmethod
    def get_labels_of(frame_predictions):
        return frame_predictions.labels
