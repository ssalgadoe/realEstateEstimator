import time
import random
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.loader import ItemLoader
from realtor.items import RealtorItem
from scrapy.http.request import Request




class RealtorSpider(scrapy.Spider):
    name = 'realtor'
   
   

    
    def start_requests(self):
        for i in range(58,60):
            url_str = "https://www.trulia.com/TX/Houston/"+str(i)+"_p/"
            print(url_str)
            yield Request(url_str, self.parse)

        
    def parse(self, response):
        print('Processing..' + response.url)
        time.sleep(random.randint(5,15))
        urls = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div/a[1]/@href").extract()
        streetAddress = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div/span[@itemprop='address']/span[@itemprop='streetAddress']/text()").extract()
        addressLoc = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div/span[@itemprop='address']/span[@itemprop='addressLocality']/text()").extract()
        addressReg = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div/span[@itemprop='address']/span[@itemprop='addressRegion']/text()").extract()
        postalCode = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div/span[@itemprop='address']/span[@itemprop='postalCode']/text()").extract()
        latitute = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div/span[@itemprop='geo']/meta[@itemprop='latitude']/@content").extract()
        longitude = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div/span[@itemprop='geo']/meta[@itemprop='longitude']/@content").extract()
        price = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div[2]/a/div[2]/div/div/span/text()").extract()
        beds = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div[2]/a/div[2]/div/div[2]/ul/li[@data-testid='beds']/text()").extract()
        baths = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div[2]/a/div[2]/div/div[2]/ul/li[@data-testid='baths']/text()").extract()
        sqft = response.xpath("//div[@class='resultsColumn']/div/div/div[@class='containerFluid']/ul/li/div/div/div[2]/a/div[2]/div/div[2]/ul/li[@data-testid='sqft']/text()").extract()
        





        
        addressLoc_ = [x for x in addressLoc  if len(x.strip()) > 0 ]
        addressReg_ = [x for x in addressReg  if len(x.strip()) > 0 ]
        print("total", len(urls), len(postalCode), len(addressLoc_))
        for index,a in enumerate(urls):
             url = response.urljoin(a)
             #print(index, streetAddress[index], addressLoc[index], addressReg[index], postalCode[index], latitute[index],longitude[index], price[index], beds[index], baths[index],sqft[index] )
             item = RealtorItem()
             item['streetAddress']  = streetAddress[index]
             item['addressLoc'] = addressLoc_[index].strip()
             item['addressReg'] = addressReg_[index]
             item['postalCode'] = postalCode[index]
             item['latitute'] = latitute[index]
             item['longitude'] = longitude[index]
             item['price'] = price[index]
             item['beds'] = beds[index]
             item['baths'] = baths[index]
             item['sqft'] = sqft[index]

#             try:
#                 yield scrapy.Request(url, callback=self.parse_details, meta = {'index':index})
#             except Exception as err:
#                 print("Erro {}".format(err))
#                 continue

             yield item
                 
    def parse_details(self, response):
        url = response.url
        print("url:", url, "index", response.meta['index'])
        try:
            address1 = response.xpath("//div[@id='propertySummary']/div/div/div/div/h1/div/text()").extract()[0].strip()
            address2 = response.xpath("//div[@id='propertySummary']/div/div/div/div/h1/span/text()").extract()[0].strip()
            print(address1, address2)
        except Exception as e:
            address1 = "n/a"     
        attrs = []
        for i in range(1,5):
            path_str = "//div[@id='propertySummary']/div/div/div/ul/li["+ str(i)+"]/text()"
            try:
                attr = response.xpath(path_str).extract()[0]
                attrs.append(attr)
#                print(attr)            
            except Exception as e:
                attr = "n/a"     



            

    def parse_details2(self, response):
        url = response.url
        try:
            address1 = response.xpath("//div[@id='propertySummary']/div/div/div/div/h1/div/text()").extract()[0].strip()
            print(address1)
        except Exception as e:
            title = "n/a"     
 
        try:        
            price = response.xpath("//div[@class='currentPrice']/div[@class='at_floatR']/span/text()").extract_first()
            if price == None:
                price = response.xpath("//span[@class='dealerPrice']/text()").extract()[0]
        except Exception as e:
            price = "n/a"    
        print("Price:", price)            
        item = AutotraderItem()
        item['Title'] = title
        item['Price'] = price
        temp = {}
        for i in range(1,10):
            try:
                key1_path = "//div[@class='specList']/div[@class='at_vehicleSpecs'][1]/div["+str(i)+"]/div[1]/span/text()"
                val1_path = "//div[@class='specList']/div[@class='at_vehicleSpecs'][1]/div["+str(i)+"]/div[2]/text()"
                alt_val1_path = "//div[@class='specList']/div[@class='at_vehicleSpecs'][1]/div["+str(i)+"]/div[2]/span/text()"

                key2_path = "//div[@class='specList']/div[@class='at_vehicleSpecs'][2]/div["+str(i)+"]/div[1]/span/text()"
                val2_path = "//div[@class='specList']/div[@class='at_vehicleSpecs'][2]/div["+str(i)+"]/div[2]/text()"
                alt_val2_path = "//div[@class='specList']/div[@class='at_vehicleSpecs'][2]/div["+str(i)+"]/div[2]/span/text()"

                key1 = response.xpath(key1_path).extract()[0].strip()
                key1=key1.replace(" ","")
                if (key1=='Style/Trim'):
                    key1 = 'Trim'
                val1 = response.xpath(val1_path).extract()[0].strip()
                if len(val1) < 1:
                    val1 = response.xpath(alt_val1_path).extract()[0].strip()

                key2 = response.xpath(key2_path).extract()[0].strip()
                key2= key2.replace(" ","")
                if (key2=='Style/Trim'):
                    key2 = 'Trim'
                
                val2 = response.xpath(val2_path).extract()[0].strip()
                if len(val2) < 1:
                    val2 = response.xpath(alt_val2_path).extract()[0].strip()


               # print("item",i,"-", "kv1:", key1,':', val1,"kv2:", key2,':', val2)
                item[key1] = val1
                item[key2] = val2
                temp[key1] = val1
                temp[key2] = val2
            except Exception as e:
                continue
        print(temp)
        yield item
#        i=2
#        key_path = "//div[@class='specList']/div[@class='at_vehicleSpecs']/div["+str(i)+"]/div[1]/span/text()"
#        val_path = "//div[@class='specList']/div[@class='at_vehicleSpecs']/div["+str(i)+"]/div[2]/text()"
#        alt_val_path = "//div[@class='specList']/div[@class='at_vehicleSpecs']/div["+str(i)+"]/div[2]/span/text()"
#        key = response.xpath(key_path).extract()[0].strip()
#        val = response.xpath(val_path).extract()[0].strip()
        
#        key = response.xpath("//div[@class='specList']/div[@class='at_vehicleSpecs']/div[2]/div[1]/span/text()").extract()[0].strip()
#        val = response.xpath("//div[@class='specList']/div[@class='at_vehicleSpecs']/div[2]/div[2]/text()").extract()[0].strip()

 
        
   
