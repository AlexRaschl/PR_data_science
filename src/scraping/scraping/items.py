# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import Field


class VideoItem(scrapy.Item):
    v_id = Field()
    v_found = Field()
    v_link = Field()
    v_title = Field()
    v_descr = Field()
    v_views = Field()
    v_filepath = Field()
    v_channel = Field()

    #   v_likes = Field()
    #   v_dislikes = Field()


    song_name = Field()
    creator = Field()
    listening_events = Field()
