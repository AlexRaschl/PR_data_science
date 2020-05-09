import glob as glob
import os
from typing import List

import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from src.config import N_SAMPLES, INDEXED_TTS_PATH, STORED_COLOR_PATH
from src.model.cfw import write_to_file
from src.preprocessing.datamanager import DataManager
from src.preprocessing.indexer import Indexer


class ColorDetector:
    def __init__(self, batch_size: int = N_SAMPLES, n_intervals: int = None):
        self.use_presets = n_intervals is None
        self.batch_size = batch_size  #
        self.borders = (self.__generate_intervals(n_intervals, prefabs=n_intervals is None))

    def get_color_aggregates(self, X: pd.DataFrame) -> pd.DataFrame:
        v_ids = DataManager.extract_v_ids(X)
        distributions = np.array([self.detect_colors(v_id) for v_id in tqdm(v_ids)])
        distributions /= distributions.sum(axis=1, keepdims=True)  # Normalizes rows to sum to 1
        if self.use_presets:
            df = pd.DataFrame(
                {'v_id': v_ids,
                 'red': distributions[:, 0],
                 'orange': distributions[:, 1],
                 'yellow': distributions[:, 2],
                 'green': distributions[:, 3],
                 'cyan': distributions[:, 4],
                 'blue': distributions[:, 5],
                 'violet': distributions[:, 6],
                 'pink': distributions[:, 7],
                 'black': distributions[:, 8],
                 'white': distributions[:, 9],
                 })
            df.set_index('v_id', inplace=True)
        else:
            df = pd.DataFrame(distributions, index=X.v_id)
        df.reset_index(inplace=True)
        return df

    def detect_colors(self, v_id: str) -> np.array:
        images = self.load_batch(v_id)
        distributions = []
        for img in images[:self.batch_size]:
            n_pixels = img.shape[0] * img.shape[1]
            color_distribution = []
            for bounds in self.borders:
                # mask = np.empty_like(img)
                if bounds[0, 0] < 0:
                    b = bounds.copy()
                    b[0, 0] = 180 + b[0, 0]
                    mask1 = cv2.inRange(img, b[0], np.array([179, b[1, 1], b[1, 2]])) / 255
                    mask2 = cv2.inRange(img, np.array([0, b[0, 1], b[0, 2]]), b[1]) / 255
                    mask = np.logical_or(mask1, mask2)
                    del bounds
                else:
                    mask = cv2.inRange(img, bounds[0], bounds[1]) / 255

                color_distribution.append(np.sum(mask) / n_pixels)
            distributions.append(color_distribution)
        return np.sum(np.array(distributions), axis=0) / len(images)

    def extract_color_features(self, v_id):
        images = self.load_batch(v_id)
        dominant_colors = []
        saturation = []
        lightness = []

        for img in images[:self.batch_size]:
            n_pixels = img.shape[0] * img.shape[1]
            color_distribution = []
            for bounds in self.borders:
                mask = cv2.inRange(img, bounds[0], bounds[1]) / 255  # TODO better intervals
                color_distribution.append(np.sum(mask) / n_pixels)
            dominant_colors.append(np.argmax(np.array(color_distribution)))

    def load_batch(self, v_id: str) -> List[np.ndarray]:
        return [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV) for path in
                natsorted(glob.glob(DataManager.sample_path_from_id(v_id) + '/*'))]

    @staticmethod
    def extract_v_ids(X: pd.DataFrame) -> List[str]:
        return X.v_id.tolist()

    def __generate_intervals(self, n_intervals: int, prefabs=True) -> np.ndarray:
        if prefabs:
            S_t = 20
            V_t = 20

            # Chosen via https://www.tydac.ch/color/
            colors = {'red': ((-25, S_t, V_t), (20, 255, 255)),
                      'orange': ((31, S_t, V_t), (45, 255, 255)),
                      'yellow': ((45, S_t, V_t), (70, 255, 255)),
                      'green': ((70, S_t, V_t), (146, 255, 255)),
                      'cyan': ((146, S_t, V_t), (191, 255, 255)),
                      'blue': ((191, S_t, V_t), (261, 255, 255)),
                      'violet': ((261, S_t, V_t), (286, 255, 255)),
                      'pink': ((286, S_t, V_t), (336, 255, 255)),
                      'black': ((0, 0, 0), (359, 255, V_t)),
                      'white': ((0, 0, V_t), (359, S_t, 255))
                      }

            borders = []
            for color, bounds in colors.items():
                lower = np.array(bounds[0])
                lower[0] /= 1
                upper = np.array(bounds[1])
                upper[0] /= 1
                borders.append(np.array([lower, upper]))

            return borders


        else:
            stepsize = int(180 / n_intervals)
            degree_bounds = list(range(0, 180, stepsize))
        return np.array([(np.array([i, 50, 50]), np.array([i + stepsize, 255, 255])) for i in degree_bounds])

    def __return_most_frequent(self, row: pd.Series) -> int:
        return row.value_counts().idxmax()


if __name__ == '__main__':
    cd = ColorDetector()
    X_train, X_test, _, _ = Indexer.load_split(folder_path=INDEXED_TTS_PATH)
    colors = cd.get_color_aggregates(X_train)
    write_to_file(os.path.join(STORED_COLOR_PATH, 'train_colors.pkl'), colors)
    colors = cd.get_color_aggregates(X_test)
    write_to_file(os.path.join(STORED_COLOR_PATH, 'test_colors.pkl'), colors)
