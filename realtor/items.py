# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class RealtorItem(scrapy.Item):
    # define the fields for your item here like:
    streetAddress = scrapy.Field()
    addressLoc = scrapy.Field()
    addressReg = scrapy.Field()
    postalCode = scrapy.Field()
    latitute = scrapy.Field()
    longitude = scrapy.Field()
    price = scrapy.Field()
    beds = scrapy.Field()
    baths = scrapy.Field()
    sqft = scrapy.Field()

    pass
