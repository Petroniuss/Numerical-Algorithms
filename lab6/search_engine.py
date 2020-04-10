import os
import numpy             as np
import scipy.sparse      as sparse
import resources_manager
from preprocessor          import preprocess_all
from resources_manager     import data_dir
from nltk.stem.porter      import PorterStemmer


### --------------------------------|
### In order to execute run:        |
###    `python search_engine.py`    |
### --------------------------------|

"""
    TODO
        - query should return k - highest ranked documents with 
            measured similarity.
        - SVD & low rank approximation
"""

k = 1

# FIXME
def query(query_text, A, terms, documents):
    q = vectorize_query(query_text, terms)
    cosines = sparse.csr_matrix.dot(q, A)

    raise Exception('Not implemented')    

def construct_term_by_document_normalized_matrix(terms, documents):
    matrix = sparse.lil_matrix(((len(terms), len(documents))))
    for i, doc in enumerate(documents):
        values = []
        for word, freq in doc.words_dict().items():
            values.append(freq)

        normalized = normalize(values)
        for j, word in enumerate(doc.words_dict().keys()):
            matrix[terms[word][0], i] = normalized[j]

    return matrix.tocsr()


def construct_term_by_document_normalized_matrix_with_IDF(terms, documents):
    N = len(documents)
    matrix = sparse.lil_matrix(((len(terms), N)), dtype = np.float32)
    
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
            # if word == 'scienc':
                # print(row_index, i, normalized[j])

    return matrix.tocsr()

# Returns normalized vector q.
def vectorize_query(query_text, terms):
    q       = np.zeros(len(terms), dtype = np.float32)
    stemmer = PorterStemmer()
    words   = query_text.split()
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


if __name__ == "__main__":
    print('Starting this super extra important task...')
    
    documents, terms = preprocess_all(data_dir)
    matrix           = construct_term_by_document_normalized_matrix_with_IDF(terms, documents)
    
    resources_manager.dump(matrix, resources_manager.matrix_path)
    matrix = resources_manager.load_dump(resources_manager.matrix_path)

    query_text = 'amateur'
    
    query(query_text, matrix, terms, documents)

    print('Done...')
