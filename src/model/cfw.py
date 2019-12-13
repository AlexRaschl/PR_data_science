import os

import numpy as np
import pandas as pd

from src.config import INDEXED_TTS_PATH, STORED_PRED_PATH, TRAIN_SIZE, TEST_SIZE
from src.preprocessing.indexer import Indexer


def write_to_file(filepath: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_pickle(filepath)


def load_from_file(filepath: str) -> pd.DataFrame:
    return pd.read_pickle(filepath)


def flattened_labels(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.applymap(lambda f: f.labels).apply(np.hstack, axis=1)


def feature_data_frame(predictions: pd.DataFrame):
    length = predictions.shape[0]
    width = predictions.shape[1] * predictions.iloc[0, 0].labels.shape[0]
    flattened = flattened_labels(predictions)

    return np.vstack(flattened.values).reshape(length, width)


def load_train_test_split():
    X_train = load_from_file(os.path.join(STORED_PRED_PATH, 'train_data.pkl'))
    X_test = load_from_file(os.path.join(STORED_PRED_PATH, 'test_data.pkl'))
    assert X_train.shape[0] == TRAIN_SIZE
    assert X_test.shape[0] == TEST_SIZE

    _, _, y_train, y_test = Indexer.load_split(INDEXED_TTS_PATH)

    return X_train, X_test, y_train.iloc[:TRAIN_SIZE], y_test.iloc[:TEST_SIZE]
