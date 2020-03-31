import glob
from typing import List

import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from src.config import *
from src.preprocessing.datamanager import DataManager
from src.preprocessing.indexer import Indexer


class FaceDetector:
    def __init__(self, batch_size=N_SAMPLES):
        self.batch_size = batch_size
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    def get_face_counts(self, X: pd.DataFrame):
        v_ids = DataManager.extract_v_ids(X)
        face_info = np.array([self.detect_faces(v_id) for v_id in tqdm(v_ids)])
        return pd.DataFrame({'v_id': v_ids,
                             'n_faces': face_info[:, 0],
                             'face_coverage': face_info[:, 1],
                             'face_per_frame': face_info[:, 2]})

    def detect_faces(self, v_id: str):
        images = self.load_batch(v_id)

        face_covered_area = 0.0
        n_faces = 0
        for img in images:
            faces = self.face_cascade.detectMultiScale(img,
                                                       scaleFactor=1.1,
                                                       minNeighbors=5,
                                                       minSize=(25, 25))

            n_faces += len(faces)
            face_covered_area += self.get_coverage(faces, img)
        return n_faces, face_covered_area / len(images), n_faces / len(images)

    def load_batch(self, v_id: str) -> List[np.ndarray]:
        return [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) for path in
                natsorted(glob.glob(DataManager.sample_path_from_id(v_id) + '/*'))]

    def get_coverage(self, faces, img):
        total_area = 0
        for (_, _, h, w) in faces:
            total_area += h * w
        return total_area / np.prod(img.shape)


if __name__ == '__main__':
    fd = FaceDetector()
    X_train, X_test, _, _ = Indexer.load_split(folder_path=INDEXED_TTS_PATH)
    face_frame = fd.get_face_counts(X_train)
    #   write_to_file(os.path.join(STORED_FACE_PATH, 'train_faces.pkl'), face_frame)
    face_frame = fd.get_face_counts(X_test)
#   write_to_file(os.path.join(STORED_FACE_PATH, 'test_faces.pkl'), face_frame)
