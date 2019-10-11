import scrapy
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor

from src.config import YT_SEARCH_STUB, CACHE_PATH
from src.extract.lfm_loader import Loader


class MusicVideoCrawler(scrapy.Spider):
    name = 'videos'
    link_extractor = LxmlLinkExtractor(allow="[.]*watch\?v=[.]*")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lfm_generator = Loader(CACHE_PATH).shuffled_list(1000)

    def start_requests(self):
        print('hello')
        for item in self.lfm_generator:
            req = scrapy.Request(url=self.__generate_search_link(item), callback=self.parse)
            yield req

    def parse(self, response):
        links = MusicVideoCrawler.link_extractor.extract_links(response)
        print(links)
        yield self.parse_search_result(response)

    def parse_search_result(self, response):
        links = MusicVideoCrawler.link_extractor.extract_links(response)
        print(links)

        yield response.body

    @staticmethod
    def __generate_search_link(lfm_data: dict) -> str:
        return f'{YT_SEARCH_STUB}{lfm_data["song_name"]} Official Video'.replace(' ', '%20')
