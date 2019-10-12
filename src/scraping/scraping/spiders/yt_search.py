import scrapy
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor

from src.config import YT_SEARCH_STUB, CACHE_PATH, YT_WATCH_STUB
from src.extract.lfm_loader import Loader


class SearchSpider(scrapy.Spider):
    name = 'search'
    link_extractor = LxmlLinkExtractor(allow="[.]*watch\?v=[.]*")
    allowed_domains = ['youtube.com']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lfm_generator = Loader(CACHE_PATH).shuffled_list(10)

    def start_requests(self):
        for item in self.lfm_generator:
            req = scrapy.Request(url=self.__generate_search_link(item), callback=self.parse)
            yield req

    def parse(self, response):
        return self.parse_search_result(response)  # Every call yields one Item with the music video found.

    def parse_search_result(self, response):
        links = SearchSpider.link_extractor.extract_links(response)
        print(links)
        for link in links:
            v_id = link.split('=')[-1]
            req = scrapy.Request(url=self.__generate_search_link(v_id), callback=self.parse_watch_request)
        return scrapy.Item()

    def parse_watch_request(self, response):
        print('hello')
        pass

    @staticmethod
    def __generate_search_link(lfm_data: dict) -> str:
        return f'{YT_SEARCH_STUB}{lfm_data["song_name"]} Official Video'.replace(' ', '%20')

    @staticmethod
    def __generate_watch_link(v_id: str):
        return f'{YT_WATCH_STUB}{v_id}'
