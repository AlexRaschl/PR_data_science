import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.database.db_utils import get_collection_from_db, CACHE_PATH


class Indexer:
    def __init__(self, seed=42):
        self.collection = get_collection_from_db()
        self.seed = seed

    def perform_train_test_split(self, folder_name: str = None, test_size: float = 0.10) -> tuple():
        """
        Create train test split of all sampled videos in the database. Returns tuple of pandas dataframes denoting test and train dataset.
        The datasets include the video_id, which can be used for sample lookups, as well as the viewcounts.
        @param folder_name: Optionally write dataframe to file
        @param test_size: Proportion of test set size
        """
        samples = list(self.collection.find(filter={'sampled': True, 'v_found': True}))
        X = pd.DataFrame(list([(s["v_id"], s['n_samples']) for s in samples]), columns=['v_id', 'n_samples'])
        y = pd.DataFrame(list([s["v_views"] for s in samples]), columns=['v_views'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.seed)

        if folder_name is not None:
            fp = os.path.join(folder_name, 'tts_' + str(self.seed))
            os.makedirs(fp, exist_ok=True)
            pd.to_pickle(X_train, os.path.join(fp, 'X_train.pkl'))
            pd.to_pickle(X_test, os.path.join(fp, 'X_test.pkl'))
            pd.to_pickle(y_train, os.path.join(fp, 'y_train.pkl'))
            pd.to_pickle(y_test, os.path.join(fp, 'y_test.pkl'))

        return X_train, X_test, y_train, y_test

    @staticmethod
    def load_split(folder_path):
        X_train = pd.read_pickle(os.path.join(folder_path, 'X_train.pkl'))
        X_test = pd.read_pickle(os.path.join(folder_path, 'X_test.pkl'))
        y_train = pd.read_pickle(os.path.join(folder_path, 'y_train.pkl'))
        y_test = pd.read_pickle(os.path.join(folder_path, 'y_test.pkl'))
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    idx = Indexer()
    idx.perform_train_test_split(CACHE_PATH)
    print(Indexer.load_split('cache/tts_42'))
