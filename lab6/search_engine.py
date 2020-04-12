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
# `python search_engine.py`    |
# --------------------------------|

# TODO create some sort of interface for performing queries

k_approx = 200
k_largest = 3


def query(query_text, A, terms, documents):
    q = vectorize_query(query_text, terms)
    cosines = sparse.csr_matrix.dot(q, A)

    docs_cosines = [(cosines[i], documents[i]) for i in range(len(documents))]

    k_matching = heapq.nlargest(
        k_largest, docs_cosines, lambda tuple: tuple[0])
    k_matching = [QueryResult(tuple[1], tuple[0]) for tuple in k_matching]

    return k_matching


def query_svd(query_text, A_k, svd, terms, documents):
    """Note that here A_k is transposed, so we dot it with q"""
    q = vectorize_query_with_IDF(query_text, terms, documents)
    q = svd.components_.dot(q)
    cosines = A_k.dot(q)

    docs_cosines = [(cosines[i], documents[i]) for i in range(len(documents))]

    k_matching = heapq.nlargest(
        k_largest, docs_cosines, lambda tuple: tuple[0])
    k_matching = [QueryResult(tuple[1], tuple[0]) for tuple in k_matching]

    return k_matching


def erm_by_document_normalized(terms, documents):
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


def vectorize_query_with_IDF(query_text, terms, documents):
    """ Returns normalized vector q. """
    q = np.zeros(len(terms), dtype=np.float32)
    N = len(documents)
    stemmer = PorterStemmer()
    words = query_text.split()
    for word in words:
        word = stemmer.stem(word)
        if word in terms:
            i = terms[word][0]
            q[i] += 1.0

    for word in set(words):
        word = stemmer.stem(word)
        if word in terms:
            i = terms[word][0]
            q[i] *= np.log(N / terms[word][1])

    return normalize(q)


def vectorize_query(query_text, terms):
    """ Returns normalized vector q. """
    q = np.zeros(len(terms), dtype=np.float32)
    stemmer = PorterStemmer()
    words = query_text.split()
    for word in words:
        word = stemmer.stem(word)
        if word in terms:
            i = terms[word][0]
            q[i] += 1.0

    return normalize(q)


def normalize(v):
    v_normalized = v
    norm = np.linalg.norm(v)
    if norm != 0:
        v_normalized = v / norm

    return v_normalized


def run_preprocessor():
    """ Note that this might take a while..."""
    resources_manager.ensure_resources()

    documents, terms = preprocess_all(data_dir)
    matrix = term_by_document_normalized_with_IDF(terms, documents)

    resources_manager.dump_sparse(matrix)
    resources_manager.dump_documents(documents)
    resources_manager.dump_terms(terms)

    A_k, svd = k_value_approximation(matrix, k_approx)
    resources_manager.dump_svd(A_k, svd)


if __name__ == "__main__":
    print('Starting this super extra important task...')
    # run_preprocessor()

    matrix = resources_manager.load_sparse()
    terms = resources_manager.load_terms()
    documents = resources_manager.load_documents()
    A_k, svd = resources_manager.load_svd()

    query_text = 'amateur'

    print(query(query_text, matrix, terms, documents))
    print(query_svd(query_text, A_k, svd, terms, documents))

    print('Done...')
