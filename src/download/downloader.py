import json
import re

from youtube_dl import YoutubeDL
from youtube_dl.utils import DownloadError

from src.config import EMPTY_PATH, DL_PATH, JSON_INFO_EXTENSION, DL_DELAY
from src.database.db_utils import get_collection_from_db

ID_regex = re.compile(r'H:\\Datasets\\YouTube\\([A-Za-z0-9_\-]{11})-')
RES_regex = re.compile(r'\(([0-9]*p)\)')
EXT_matcher = re.compile(r'(.[A-Za-z0-9]*$)')


class Downloader:

    def __init__(self):
        self.collection = get_collection_from_db()
        self.options = {
            # 'format': 'bestaudio/best',
            'outtmpl': f'{DL_PATH}%(id)s-%(title)s-%(format)s.%(ext)s',
            'noplaylist': True,
            # progress_hooks': [my_hook],
            # 'simulate': True,
            'writeinfojson': True,
            # 'download_archive': DL_PATH,
            'include_ads': False,
            'call_home': False,
            'sleep_interval': DL_DELAY,  # TODO extract as constant
            'progress_hooks': [self.dl_hook],
            'format': 'best[height<=360]'  # Download in 360p or lower
        }
        self.ydl = YoutubeDL(self.options)

    def download(self, num=1000, delay=10, dl_only_mv=True):
        # Find entries which have not been downloaded yet
        doc = self.collection.find({"$and": [{'v_found': dl_only_mv}, {'v_filepath': {"$eq": EMPTY_PATH}}]})
        for vid in doc:
            try:
                self.ydl.download([vid['v_link']])
            except DownloadError as de:
                print(de)
                with open('dl_error.log', 'a') as f:
                    print('Could not download video. Required resolution not available!!',
                          file=f)  # TODO maybe mark this in database
                    print(vid['v_id'], file=f)
                    print('\n\n', file=f)

        # doc = self.collection.find_one({"$and": [{'v_found': dl_only_mv}, {'v_filepath': {"$eq": EMPTY_PATH}}]})
        # try:
        #     self.ydl.download([doc['v_link']])
        # except DownloadError as de:
        #     print(de)
        #     with open('dl_error.log', 'a') as f:
        #         print('Could not download video. Required resolution not available!!', file=f) # TODO maybe mark this in database
        #         print(doc['v_id'], file=f)
        #         print('\n\n', file=f)

    def dl_hook(self, d):
        print(d)
        if d['status'] == 'finished':
            fname = d.get('filename')
            if fname is not None:
                print(f'Finished downloading:{fname}. Now updating the DB.')
                extension = EXT_matcher.search(fname).group(1)
                with open(fname.replace(extension, JSON_INFO_EXTENSION), 'r') as json_file:
                    info = json.load(json_file)
                    self.collection.update_one({'v_id': extract_id(fname)},
                                               {"$set": {
                                                   'v_filepath': fname.replace(DL_PATH, ''),
                                                   'v_res': RES_regex.search(fname).group(1),
                                                   'v_likes': info['like_count'],
                                                   'v_dislikes': info['dislike_count'],
                                                   'v_avg_rating': info['average_rating']
                                               }
                                               })
            else:
                raise ValueError("filename not in download hook parameter!")
        if d['status'] == 'downloading':
            print(d['filename'], d['_percent_str'], d['_eta_str'])


def extract_id(filepath):
    return ID_regex.search(filepath).group(1)


if __name__ == '__main__':
    dl = Downloader()
    dl.download()
