import pymongo
import scrapy
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor

from src.config import YT_SEARCH_STUB, CACHE_PATH, YT_WATCH_STUB
from src.extract.lfm_loader import Loader
from src.scraping.scraping.items import VideoItem
from src.scraping.scraping.util import VideoInspector


class SearchSpider(scrapy.Spider):
    name = 'search'
    link_extractor = LxmlLinkExtractor(allow="[.]*watch\?v=[.]*")
    allowed_domains = ['youtube.com']

    inspector = VideoInspector()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lfm_generator = Loader(CACHE_PATH).shuffled_list(1000)
        self.collection = self.get_collection_from_db()


    def start_requests(self):
        for item in self.lfm_generator:
            if not self.collection.find_one({'song_name': item['song_name']}):
                req = scrapy.Request(url=self.__generate_search_link(item), callback=self.parse_search_result,
                                     cb_kwargs=item)
                yield req

    def parse(self, response):
        return self.parse_search_result(response)  # Every call yields one Item with the music video found.

    def parse_search_result(self, response, song_name, creator, listening_events):

        # Extract html <a> containing v_id and title info
        vid_a = response.selector.xpath(
            r'//li/div/div/div[contains(@class, yt-lockup-content)]/h3/a[contains(@href,"watch?v")]').get()

        # Extract meta <div> containing number of views
        # vid_meta = response.selector.xpath('//li/div/div/div[contains(@class, yt-lockup-content)]/span').get()

        # Extract html <div> snippet of description
        vid_descr = response.selector.xpath(
            r'//li/div/div/div[contains(@class, yt-lockup-content)]/div[contains(@class,yt-lockup-description) and @dir="ltr"]').get()

        # Extract all the above
        info = self.inspector.extract(vid_a, vid_descr)
        # print(info)

        if self.inspector.is_music_video(info, song_name, creator):
            # TODO make new request to fetch all data (parse_watch_request)
            return self.__generate_item(info, song_name, creator, listening_events)

        print(f"Found no video for: {song_name}")  # TODO log this in file

        return

    def parse_watch_request(self, response):

        pass

    @staticmethod
    def __generate_search_link(lfm_data: dict) -> str:
        return f'{YT_SEARCH_STUB}{lfm_data["song_name"]} Official Video'.replace(' ', '%20')

    @staticmethod
    def __generate_watch_link(v_id: str):
        return f'{YT_WATCH_STUB}{v_id}'

    @staticmethod
    def __generate_item(info, song_name, creator, listening_events):
        return VideoItem(v_id=info['v_id'], v_link=YT_WATCH_STUB + info['v_id'], v_title=info['title'],
                         v_descr=info['descr'], song_name=song_name, creator=creator, listening_events=listening_events)

    def get_collection_from_db(self):
        conn = pymongo.MongoClient(
            'localhost',
            27017
        )
        db = conn['musicvideos']
        return db['video_info']
