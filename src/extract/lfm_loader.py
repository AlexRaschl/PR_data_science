import os

import pandas as pd


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

    def next_shuffled_list_item(self):
        data = self.load_data()

        for i in range(0, int(data.shape[0] / 2)):
            print(i)
            yield data.iloc[i]
            yield data.iloc[-i - 1]

        if data.shape[0] % 2 != 0:
            yield data.iloc[int(data.shape[0] / 2)]


if __name__ == '__main__':
    loader = Loader('cache')
    shuffler = loader.next_shuffled_list_item()
    for i in range(0, 10):
        print(f'{next(shuffler)}\n')
