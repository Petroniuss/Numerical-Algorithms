import os
import numpy as np
import scipy.sparse as sparse
from preprocessor import preprocess_all
from resources_manger import data_dir


### --------------------------------|
### In order to execute run:        |
###    `python search_engine.py`    |
### --------------------------------|

def construct_term_by_document_matrix(terms, documents):
    matrix = sparse.lil_matrix(((len(terms), len(documents))))
    for i, doc in enumerate(documents):
        for word, freq in doc.words_dict().items():
            matrix[terms[word][0], i] = freq

    return matrix.tocsr()


def construct_term_by_document_matrix_with_IDF(terms, documents):
    N = len(documents)
    matrix = sparse.lil_matrix(((len(terms), N)), dtype = np.float32)
    
    for i, doc in enumerate(documents):
        for word, freq in doc.words_dict().items():
            nw = terms[word][1]
            matrix[terms[word][0], i] = freq * np.log(N / nw)

    return matrix.tocsr()


if __name__ == "__main__":
    print('Starting this super extra important task...')
    
    documents, terms = preprocess_all(data_dir)
    matrix           = construct_term_by_document_matrix_with_IDF(terms, documents)
    print(matrix.size)

    print('Done...')