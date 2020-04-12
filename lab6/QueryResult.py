class QueryResult:

    def __init__(self, document, similarity_value):
        self.document = document
        self.similarity_measure = similarity_value

    def __str__(self):
        document = str(self.document)
        line = (f'Similarity - {self.similarity_measure:.2f}').center(26)
        footer = '=' * 26

        return document + line + '\n' + footer + '\n'

    def __repr__(self):
        return str(self)
