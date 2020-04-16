import os
import numpy as np
import scipy.sparse as sparse
import resources_manager
import heapq
from QueryResult import QueryResult
from preprocessor import preprocess_all
from resources_manager import data_dir
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD


# --------------------------------|
# In order to execute run:        |
#     `python SearchEngine.py`   |
# --------------------------------|

k_approx = 200


class SearchEngine:
    """ Client api """

    def __init__(self, run_preprocessor=False):
        if run_preprocessor:
            preprocessor()

        self.A = resources_manager.load_sparse()
        self.terms = resources_manager.load_terms()
        self.documents = resources_manager.load_documents()
        self.A_k, self.svd = resources_manager.load_svd()

    def recalcuclateSVD(self, k):
        self.A_k, self.svd = k_value_approximation(self.A, k)
        resources_manager.dump_svd(self.A_k, self.svd)

    def query(self, query_text, use_svd=True, k_largest=20):
        q = self.vectorize_query(query_text)
        cosines = None
        if use_svd:
            q = self.svd.components_.dot(q)
            cosines = self.A_k.dot(q)
        else:
            cosines = sparse.csr_matrix.dot(q, self.A)

        docs_cosines = [(cosines[i], self.documents[i])
                        for i in range(len(self.documents))]

        k_matching = heapq.nlargest(
            k_largest, docs_cosines, lambda tuple: tuple[0])
        results = [QueryResult(tuple[1], tuple[0]) for tuple in k_matching]

        return results

    def vectorize_query(self, query_text):
        """ Returns normalized vector q, with IDF"""

        q = np.zeros(len(self.terms), dtype=np.float32)
        stemmer = PorterStemmer()
        words = query_text.split()
        N = len(self.documents)

        for word in words:
            word = stemmer.stem(word)
            if word in self.terms:
                i = self.terms[word][0]
                q[i] += 1.0

        for word in set(words):
            word = stemmer.stem(word)
            if word in self.terms:
                i = self.terms[word][0]
                q[i] *= np.log(N / self.terms[word][1])

        return normalize(q)


# ----------------------------------------------------
# Code below is used only afrer initial preprocessing.
# ----------------------------------------------------


def preprocessor():
    """ Note that this might take a while..."""
    resources_manager.ensure_resources()

    documents, terms = preprocess_all(data_dir)
    matrix = term_by_document_normalized_with_IDF(terms, documents)
    # matrix = term_by_document_normalized(terms, documents)

    resources_manager.dump_sparse(matrix)
    resources_manager.dump_documents(documents)
    resources_manager.dump_terms(terms)

    A_k, svd = k_value_approximation(matrix, k_approx)
    resources_manager.dump_svd(A_k, svd)


def term_by_document_normalized(terms, documents):
    matrix = sparse.lil_matrix(((len(terms), len(documents))))
    for i, doc in enumerate(documents):
        values = []
        for word, freq in doc.words_dict().items():
            values.append(freq)

        normalized = normalize(values)
        for j, word in enumerate(doc.words_dict().keys()):
            matrix[terms[word][0], i] = normalized[j]

    return matrix.tocsr()


def term_by_document_normalized_with_IDF(terms, documents):
    N = len(documents)
    matrix = sparse.lil_matrix(((len(terms), N)), dtype=np.float32)

    for i, doc in enumerate(documents):
        values = []
        for word, freq in doc.words_dict().items():
            nw = terms[word][1]
            value = freq * np.log(N / nw)
            values.append(value)

        normalized = normalize(values)
        for j, word in enumerate(doc.words_dict().keys()):
            row_index = terms[word][0]
            matrix[row_index, i] = normalized[j]

    return matrix.tocsr()


def k_value_approximation(A, k):
    svd = TruncatedSVD(n_components=k).fit(A.T)
    A_k = svd.transform(A.T)

    return A_k, svd


def normalize(v):
    v_normalized = v
    norm = np.linalg.norm(v)
    if norm != 0:
        v_normalized = v / norm

    return v_normalized


if __name__ == "__main__":
    print('Starting this super extra important task...')

    SearchEngine(run_preprocessor=True)

    print('Done...')
