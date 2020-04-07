import scrapy

main_url = 'https://en.wikipedia.org/'

### --------------------------------|
### In order to execute run:        |
###    `scrapy runspider scraper.py`|
### --------------------------------|
class WikiSpider(scrapy.Spider):
    name = "wiki_spider"
    start_urls = [main_url + 'wiki/Mathematics']

    num_to_download = int(1.5 * 1e4)
    set = set(start_urls[0])

    def parse(self, response):
        page = response.url.split('/')[-1]

        filename = 'docs/{}.html'.format(page)
        with open(filename, 'wb') as file:
            file.write(response.body)

        if len(self.set) >= self.num_to_download:
            return

        for link in response.css('a::attr(href)').getall():
            nodes = link.split('/')
            if len(nodes) > 2 and nodes[1] == 'wiki' and ':' not in link and 'disambiguation' not in link:
                next_page = main_url + link
                if link not in self.set and len(self.set) < self.num_to_download:
                    self.set.add(link)
                    yield scrapy.Request(next_page, callback = self.parse)