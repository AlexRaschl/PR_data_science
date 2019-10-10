import json

from googleapiclient.discovery import build

from src.config import API_KEY


class Crawler:

    def __init__(self):
        self.yt_service = build('youtube', 'v3', developerKey=API_KEY)

    def search_for_music_video(self, song_name: str, creator: str):
        """
        Searches for the official music video of a song given the song name and the creator of the song.
        :param song_name: name of the song
        :param creator: creator of the song
        :return: JSON like dict of information including link, views, likes....
        """
        req = self.yt_service.search().list(q=f'{song_name} Official Video', part='snippet', type='video')
        return req.execute()['items']

    def search_list(self, path_to_list: str, shuffle: bool = True):
        """
        Yields dicts for storing data in the db
        :param path_to_list:
        :param shuffle:%
        :return:
        """
        pass


if __name__ == '__main__':
    c = Crawler()
    print(json.dumps(c.search_for_music_video('Somebody that i used to know', 'Gotye'), indent=2))
