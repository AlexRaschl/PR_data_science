import os

import pandas as pd

from src.config import CACHE_PATH


class Loader:
    def __init__(self, root_path, read_path: str = 'data/LFM-1b.txt'):
        self.root = root_path  # Path for the cache files
        self.read_path = read_path

        os.makedirs(self.root, exist_ok=True)

        self.data_path = os.path.join(self.root, 'lfm_1b.pickle')

        if not self.is_already_cached():
            self.data = pd.read_csv(self.read_path, sep=';', header=None)

            if self.data is None or self.data.empty:
                raise ValueError('Error: No data loaded!')

            _ = self.load_data()

            del self.data

    def is_already_cached(self):
        return os.path.exists(self.data_path)

    def load_data(self):
        if os.path.exists(self.data_path):
            return pd.read_pickle(self.data_path)
        else:
            pd.to_pickle(self.data, self.data_path)
            return self.data

    def shuffled_list(self, n_items: int = 0):
        data = self.load_data()
        size = data.shape[0] if n_items == 0 else n_items

        for i in range(0, int(size / 2)):
            yield self.convert_to_dict(data.iloc[i])
            yield self.convert_to_dict(data.iloc[-i - 1])

        if data.shape[0] % 2 != 0:
            yield self.convert_to_dict(data.iloc[int(data.shape[0] / 2)])

    def convert_to_dict(self, list_item):
        return {
            'song_name': list_item[0],
            'creator': list_item[1],
            'listening_events': str(list_item[2])  # Convert to string for latter json compatibility
        }

if __name__ == '__main__':
    loader = Loader(CACHE_PATH)
    shuffler = loader.shuffled_list(10)
    for i in range(0, 10):
        print(f'{next(shuffler)}\n')
