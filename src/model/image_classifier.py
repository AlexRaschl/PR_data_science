import os
from itertools import product
from itertools import product
from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd
from natsort import natsorted
from tensorflow.keras import Input
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import DL_PATH, N_SAMPLES, TRAIN_SIZE, TEST_SIZE
from src.preprocessing.indexer import Indexer
from src.preprocessing.wrappers import FramePredictions


class ImageClassifier:

    def __init__(self, batch_size: int = N_SAMPLES, n_predictions=5):
        self.batch_size = batch_size
        self.input_shape = None
        self.model = None
        self.n_predictions = n_predictions

    def init_model(self, model_name: str = 'VGG16'):
        if model_name == 'VGG16':
            self.input_shape = Input(shape=(224, 224, 3))
            self.model = VGG16(weights='imagenet')

        elif model_name == 'ResNet50v2':
            self.input_shape = shape = (224, 224, 3)
            self.model = ResNet50V2(weights='imagenet')

    def classify(self, X):

        predictions = np.empty(shape=(X.shape[0], self.batch_size),
                               dtype=np.dtype([
                                   ('predictions', np.int64, (self.n_predictions,)),
                                   ('probabilities', np.float64, (self.n_predictions,))]))

        for idx, v_id in enumerate(X.iloc[:, 0]):
            batch = self.load_batch(v_id)
            res = self.model.predict(batch, batch_size=self.batch_size)
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
        del predictions
        return df

    def load_batch(self, v_id: str):
        root = join(DL_PATH, v_id)
        fnames = {'filenames': natsorted([f for f in listdir(root) if isfile(join(root, f))])}
        df = pd.DataFrame(fnames)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False)

        train_generator = train_datagen.flow_from_dataframe(
            df,
            directory=join(DL_PATH, v_id),
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode=None,
            x_col='filenames')

        return train_generator.next()


if __name__ == '__main__':
    img_cf = ImageClassifier()
    X_train, X_test, _, _ = Indexer.load_split(folder_path='cache/tts_42')
    print(X_train.shape)
    print(X_test.shape)
    img_cf.init_model(model_name='ResNet50v2')
    classifications = img_cf.classify(X_train.iloc[:TRAIN_SIZE])
    write_to_file(os.path.join(STORED_PRED_PATH, 'train_data.pkl'), classifications)

    classifications = img_cf.classify(X_test.iloc[:TEST_SIZE])
    write_to_file(os.path.join(STORED_PRED_PATH, 'test_data.pkl'), classifications)
