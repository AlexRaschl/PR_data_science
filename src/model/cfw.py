import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.config import INDEXED_TTS_PATH, STORED_PRED_PATH, TRAIN_SIZE, TEST_SIZE, SPLIT_SEED
from src.preprocessing.indexer import Indexer


def write_to_file(filepath: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_pickle(filepath)


def load_from_file(filepath: str) -> pd.DataFrame:
    return pd.read_pickle(filepath)


def flattened_labels(predictions: pd.DataFrame, n_labels=None) -> pd.DataFrame:
    n_labels = _fetch_n_labels(predictions, n_labels)

    return predictions.applymap(lambda f: f.labels[0:n_labels]).apply(np.hstack, axis=1)


def feature_data_frame(predictions: pd.DataFrame, n_labels=None):
    n_labels = _fetch_n_labels(predictions, n_labels)
    length, width = _fetch_len_width(predictions, n_labels)

    flattened = flattened_labels(predictions, n_labels)

    return pd.DataFrame(np.vstack(flattened.values).reshape(length, width), index=predictions.index)


def get_textual_description(predictions: pd.DataFrame, selector=None, n_labels=None):
    if selector is None:
        return label_to_text_data_frame(predictions, n_labels)
    elif isinstance(selector, (int, tuple)):
        return label_to_text_data_frame(predictions.iloc[selector], n_labels)


def label_to_text_data_frame(predictions: pd.DataFrame, n_labels=None):
    n_labels = _fetch_n_labels(predictions, n_labels)
    length, width = _fetch_len_width(predictions, n_labels)
    flattened = predictions.applymap(lambda f: f.return_label_descr()[0:n_labels])
    flattened = flattened.apply(np.hstack, axis=1)
    return pd.DataFrame(np.vstack(flattened.values).reshape(length, width), index=predictions.index)


def load_train_test_split(dataset: str = 'CNN', split_seed: int = SPLIT_SEED,
                          feature_frame=False, n_labels=None, **kwargs):
    # TODO INCORPORATE SPLIT SEED
    if dataset == 'CNN':
        X_train = load_from_file(os.path.join(STORED_PRED_PATH, 'train_cnn_agg_False.pkl'))
        X_test = load_from_file(os.path.join(STORED_PRED_PATH, 'test_cnn_agg_False.pkl'))

        _, _, y_train, y_test = Indexer.load_split(INDEXED_TTS_PATH)

        _check_shapes(X_train, X_test)
        _check_shapes(y_train, y_test)

        y_train.set_index(X_train.index, inplace=True)
        y_test.set_index(X_test.index, inplace=True)

        if feature_frame:
            X_train = feature_data_frame(X_train, n_labels)
            X_test = feature_data_frame(X_test, n_labels)

            if kwargs.get('one_hot_inputs', False):
                categories = list([range(0, 1000) for i in range(0, X_train.shape[1])])
                ohe = OneHotEncoder(categories=categories, dtype=np.int64)
                ohe.fit(X_train)
                X_train = ohe.transform(X_train)
                X_test = ohe.transform(X_test)

        return X_train, X_test, y_train, y_test
    else:
        raise NotImplementedError('Other tts not implemented yet!')


def _fetch_n_labels(predictions: pd.DataFrame, n_labels=None) -> int:
    max_labels = predictions.iloc[0, 0].labels.shape[0]
    return n_labels if n_labels else max_labels


def _fetch_len_width(predictions: pd.DataFrame, n_labels) -> tuple:
    return predictions.shape[0], predictions.shape[1] * n_labels


def _check_shapes(train: pd.DataFrame, test: pd.DataFrame):
    if train.shape[0] != TRAIN_SIZE or test.shape[0] != TEST_SIZE:
        raise ValueError('Shapes do not match!')


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true) * 100)


def mean_relative_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred / y_true))
