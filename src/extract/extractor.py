import json

from googleapiclient.discovery import build

from src.config import API_KEY, YT_WATCH_STUB


class YT_API_Crawler:

    def __init__(self):
        self.yt_service = build('youtube', 'v3', developerKey=API_KEY)

    def search_for_music_video(self, lfm_data: dict) -> dict:
        """
        Searches for the official music video of a song given the song name and the creator of the song.
        :param lfm_data: Dictionary consisting of song_name, creator, listening_events (based on lfm-1b dataset)
        :return: dict of information including link, views, likes....
        """
        req = self.yt_service.search().list(q=f'{lfm_data["song_name"]} Official Video', part='snippet', type='video')
        for item in req.execute()['items']:
            if self.is_music_video(item, lfm_data['song_name'], lfm_data['creator']):
                item, stats = self.extract_item_info(item)
                return self.construct_json_dict(item, stats, lfm_data)
        return lfm_data  # Return lfm data if nothing found

    def search_list(self, path_to_list: str, shuffle: bool = True):
        """
        Yields dicts for storing data in the db
        :param path_to_list:
        :param shuffle:
        :return:
        """
        pass

    def is_music_video(self, search_item: dict, song_name: str, creator: str) -> bool:
        """
        Checks if the found YouTube video is indeed a music-video.
        Compares Video title with `song_name` and compares the channel name with the `creator` of the song.
        Furthermore, it searches for the string 'Official Video' inside the video description and title.

        :param search_item:
        :param song_name:
        :param creator:
        :return: `True` if it is believed that the video is indeed a valid music video else `False`
        """
        # TODO: Hard fact-> Song title in video title
        # TODO: Hard fact -> 'Lyrics' not in video title
        # TODO: Hard fact -> 'Fan-vid(eo)' not in video description
        # TODO: Soft hint -> video in title
        # TODO: Soft hint -> creator name in channel name
        # TODO: Soft hint -> Official music video in description
        # TODO: Hard fact -> Cover not in title

        return True

    def extract_item_info(self, item: dict):
        v_id = item['id']['videoId']
        item = self.yt_service.videos().list(id=v_id, part='snippet').execute()['items'][0]
        statistics = self.yt_service.videos().list(id=v_id, part='statistics').execute()['items'][0]['statistics']
        return item, statistics

    @staticmethod
    def construct_json_dict(item: dict, statistics: dict, lfm_data: dict):

        if not item or not statistics or not lfm_data:
            raise ValueError('Cannot construct json-dict from None!')
        try:
            del item['snippet']['thumbnails']
        except KeyError:
            print(f'Key not found: snippet.thumbnails: {item}')
        try:
            del item['snippet']['tags']
        except KeyError:
            print(f'Key not found: snippet.tags: {item}')
        try:
            del item['snippet']['liveBroadcastContent']
        except KeyError:
            print(f'Key not found: snippet.liveBroadcastContent: {item}')
        try:
            del item['snippet']['localized']
        except KeyError:
            print(f'Key not found: snippet.localized: {item}')

        item['link'] = YT_WATCH_STUB + item['id']
        item['statistics'] = statistics
        item['lfm_data'] = lfm_data
        return item


def test_YT_Crawler():
    c = YT_API_Crawler()
    r = c.search_for_music_video(
        {
            'song_name': 'We Are Young (feat. Janelle Monáe)',
            'creator': 'fun.',
            'listening_events': 1337
        }
    )
    print(json.dumps(r, indent=2))



if __name__ == '__main__':
    pass
    # test_YT_Crawler()
