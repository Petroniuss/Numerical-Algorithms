import scrapy
import os

main_url = 'https://en.wikipedia.org/'
save_to = os.path.join(os.getcwd(), 'resources', 'data')
num_to_download = 10000

# --------------------------------|
# In order to execute run:        |
# `scrapy runspider scraper.py`|
# --------------------------------|


class WikiSpider(scrapy.Spider):
    name = "wiki_spider"
    start_urls = [main_url + 'wiki/Science']

    links = set([start_urls[0]])

    def parse(self, response):
        page = response.url.split('/')[-1]

        filename = '{}.html'.format(page)
        full_filename = os.path.join(save_to, filename)
        with open(full_filename, 'wb') as file:
            file.write(response.body)

        if len(self.links) >= num_to_download:
            return

        for link in response.css('a::attr(href)').getall():
            nodes = link.split('/')
            if len(nodes) > 2 and nodes[1] == 'wiki' and \
               ':' not in link and \
               '%' not in link and \
               'disambiguation' not in link:

                next_page = main_url + link
                if link not in self.links and len(self.links) < num_to_download:
                    self.links.add(link)
                    yield scrapy.Request(next_page, callback=self.parse)
