from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd
from natsort import natsorted
from tensorflow.keras import Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import DL_PATH, N_SAMPLES
from src.preprocessing.indexer import Indexer


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

    def classify(self, X):
        y_pred = pd.DataFrame(X.iloc[:, 0])
        y_pred.columns = ['v_id']
        y_pred.set_index('v_id', inplace=True)
        predictions = np.empty(shape=(y_pred.shape[0], self.batch_size),
                               dtype=np.dtype([
                                   ('predictions', np.int64, (self.n_predictions,)),
                                   ('probabilities', np.float64, (self.n_predictions,))]))

        for i in range(self.batch_size):
            y_pred['predictions_%d' % i] = None

        for v_id in X.iloc[:, 0]:
            batch = self.load_batch(v_id)
            res = self.model.predict(batch, batch_size=self.batch_size)
            i = 0
            v_idx = y_pred.index.get_loc(v_id)
            labels = np.argsort(res, axis=-1)[:, -self.n_predictions - 1:-1]
            labels = labels[:, ::-1]
            probabilities = np.empty(shape=(30, 5), dtype=np.float64)
            for i in range(30):
                probabilities = res[i, labels[i]]

            predictions[v_idx, :]['predictions'] = labels
            predictions[v_idx, :]['probabilities'] = probabilities
        return predictions

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
    X_train, X_test, y_train, y_test = Indexer.load_split(folder_path='cache/tts_42')
    y_pred = X_train.iloc[:, 0]
    y_pred.columns = ['v_id']

    img_cf.init_model()
    res = img_cf.classify(X_train.iloc[:10])
    print(res)
