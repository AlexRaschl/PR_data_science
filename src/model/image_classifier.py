import os
from itertools import product
from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd
from natsort import natsorted
from tensorflow.keras import Input
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from src.config import DL_PATH, N_SAMPLES, INDEXED_TTS_PATH, STORED_PRED_PATH
from src.model.cfw import write_to_file
from src.preprocessing.indexer import Indexer
from src.preprocessing.wrappers import FramePredictions


class ImageClassifier:

    def __init__(self, batch_size: int = N_SAMPLES, n_predictions=10):
        self.batch_size = batch_size
        self.input_shape = None
        self.model = None
        self.n_predictions = n_predictions

    def init_model(self, model_name: str = 'VGG16'):
        if model_name == 'VGG16':
            self.input_shape = Input(shape=(224, 224, 3))
            self.model = VGG16(weights='imagenet')

        elif model_name == 'ResNet50':
            self.input_shape = (224, 224, 3)
            self.model = ResNet50(weights='imagenet')

    def classify(self, X: pd.DataFrame, aggregate: bool = False) -> pd.DataFrame:
        return self.__compute_aggregated_predictions(X) if aggregate \
            else self.__compute_full_predictions(X)

    def __compute_full_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        predictions = np.empty(shape=(X.shape[0], self.batch_size),
                               dtype=np.dtype([
                                   ('predictions', np.int64, (self.n_predictions,)),
                                   ('probabilities', np.float64, (self.n_predictions,))]))

        for idx, v_id in enumerate(X.iloc[:, 0]):
            batch = self.load_batch(v_id)
            res = self.model.predict(batch, batch_size=self.batch_size)
            # pr = decode_predictions(res, top=5)
            labels = np.argsort(res, axis=-1)[:, -self.n_predictions - 1:-1]
            labels = labels[:, ::-1]
            probabilities = np.empty(shape=(30, 5), dtype=np.float64)
            print(f'Prediction step: {idx}')
            for i in range(labels.shape[0]):
                probabilities = res[i, labels[i]]

            predictions[idx, :labels.shape[0]]['predictions'] = labels
            predictions[idx, :labels.shape[0]]['probabilities'] = probabilities

        wrappers = []

        for i, (r, c) in enumerate(product(range(X.shape[0]), range(self.batch_size))):
            wrappers.append(FramePredictions(
                predictions[r, c]['predictions'],
                predictions[r, c]['probabilities']
            ))
        wrappers = np.asarray(wrappers, dtype=type(FramePredictions)).reshape(X.shape[0], self.batch_size)

        df = pd.DataFrame(wrappers).set_index(pd.Index(X.iloc[:, 0]))
        df.columns = [f'frame_{s}' for s in range(self.batch_size)]
        df.reset_index(inplace=True)
        del predictions
        return df

    def __compute_aggregated_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        data = np.empty(shape=(X.shape[0], 1000))

        for idx, v_id in enumerate(tqdm(X.iloc[:, 0])):
            batch = self.load_batch(v_id)
            predictions = self.model.predict(batch, batch_size=self.batch_size)
            predictions = predictions.sum(axis=0)  # Sum all frames
            predictions = predictions / predictions.sum(axis=0)  # Normalize probabilities
            data[idx] = predictions

        df = pd.DataFrame(data=data, index=X.iloc[:, 0])
        df.reset_index(inplace=True)
        return df

    def load_batch(self, v_id: str):
        root = join(DL_PATH, v_id)
        fnames = {'filenames': natsorted([f for f in listdir(root) if isfile(join(root, f))])}
        df = pd.DataFrame(fnames)

        datagen = ImageDataGenerator()
        generator = datagen.flow_from_dataframe(df, batch_size=self.batch_size,
                                                directory=join(DL_PATH, v_id),
                                                shuffle=False, x_col='filenames',
                                                target_size=(224, 224),
                                                class_mode=None)

        return preprocess_input(generator.next())


if __name__ == '__main__':
    img_cf = ImageClassifier()
    X_train, X_test, _, _ = Indexer.load_split(folder_path=INDEXED_TTS_PATH)
    print(X_train.shape)
    print(X_test.shape)
    img_cf.init_model(model_name='ResNet50')

    aggregate = False

    classifications = img_cf.classify(X_train, aggregate)
    write_to_file(os.path.join(STORED_PRED_PATH, f'train_cnn_agg_{str(aggregate)}.pkl'), classifications)

    classifications = img_cf.classify(X_test, aggregate)
    write_to_file(os.path.join(STORED_PRED_PATH, f'test_cnn_agg_{str(aggregate)}.pkl'), classifications)
