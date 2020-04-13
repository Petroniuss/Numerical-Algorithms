import json
import QueryResult


class QueryResponse:

    def __init__(self, article, link, similarity):
        self.link = link.replace('\\', '\\\\')
        self.article = article.replace('_', ' ')
        self.similarity = similarity

    def __repr__(self):
        return """{{
                    "article": "{}",
                    "link": "{}",
                    "similarity": {:.5f}
                  }} 
               """.format(self.article, self.link, self.similarity)


def from_query_result(query_result: QueryResult):
    similarity = query_result.similarity_measure
    link = query_result.document.link()
    article = query_result.document.filename.replace('.html', '')

    return QueryResponse(article, link, similarity)
