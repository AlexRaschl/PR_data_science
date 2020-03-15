import glob
import os
from typing import List, Tuple

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
    def __init__(self, batch_size: int = N_SAMPLES, n_intervals: int = 10):
        self.batch_size = batch_size
        self.borders = (self.__generate_intervals(n_intervals))

    def get_color_aggregates(self, X: pd.DataFrame) -> pd.DataFrame:
        v_ids = DataManager.extract_v_ids(X)
        dominant_colors = np.array([self.detect_colors(v_id) for v_id in tqdm(v_ids)])
        df = pd.DataFrame(dominant_colors, index=X.v_id)
        df.columns = [f'frame_{i}' for i in range(dominant_colors.shape[1])]
        df.reset_index(inplace=True)
        df['most_dominant'] = df.apply(lambda row: self.__return_most_frequent(row), axis=1)
        return df

    def detect_colors(self, v_id: str) -> List[np.ndarray]:
        images = self.load_batch(v_id)
        dominant_colors = []
        for img in images:
            n_pixels = img.shape[0] * img.shape[1]
            color_distribution = []
            for bounds in self.borders:
                mask = cv2.inRange(img, bounds[0], bounds[1]) / 255  # TODO better intervals
                color_distribution.append(np.sum(mask) / n_pixels)
            dominant_colors.append(np.argmax(np.array(color_distribution)))
        return dominant_colors

    def load_batch(self, v_id: str) -> List[np.ndarray]:
        return [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV) for path in
                natsorted(glob.glob(DataManager.sample_path_from_id(v_id) + '/*'))]

    @staticmethod
    def extract_v_ids(X: pd.DataFrame) -> List[str]:
        return X.v_id.tolist()

    def __generate_intervals(self, n_intervals: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        stepsize = int(180 / n_intervals)
        degree_bounds = list(range(0, 180, stepsize))
        return [(np.array([i, 50, 50]), np.array([i + stepsize, 255, 255])) for i in degree_bounds]

    def __return_most_frequent(self, row: pd.Series) -> int:
        return row.value_counts().idxmax()


if __name__ == '__main__':
    cd = ColorDetector()
    X_train, X_test, _, _ = Indexer.load_split(folder_path=INDEXED_TTS_PATH)
    colors = cd.get_color_aggregates(X_train)
    write_to_file(os.path.join(STORED_COLOR_PATH, 'train_colors.pkl'), colors)
    colors = cd.get_color_aggregates(X_test)
    write_to_file(os.path.join(STORED_COLOR_PATH, 'test_colors.pkl'), colors)
