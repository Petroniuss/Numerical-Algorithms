import pickle
import nltk
import os

resources_path = os.path.join(os.getcwd(), 'resources')
data_dir = os.path.join(resources_path, 'data')
processed_dir = os.path.join(resources_path, 'processed')
processed_data_dir = os.path.join(processed_dir, 'data')

sparse_dump_path = os.path.join(processed_dir, 'sparse.pickle')
documents_dump_path = os.path.join(processed_dir, 'docs.pickle')
terms_dump_path = os.path.join(processed_dir, 'terms.pickle')
k_value_approx_dump_path = os.path.join(processed_dir, 'k_value_approx.pickle')
svd_dump_path = os.path.join(processed_dir, 'svd.pickle')

# -------------------------------------|
# Contains utilties:                   |
# Saving & loading proccessed data`|
# -------------------------------------|


def load_svd():
    svd = load_dump(svd_dump_path)
    A_k = load_dump(k_value_approx_dump_path)

    return A_k, svd


def dump_svd(A_k, svd):
    dump(A_k, k_value_approx_dump_path)
    dump(svd, svd_dump_path)


def load_sparse():
    return load_dump(sparse_dump_path)


def load_documents():
    return load_dump(documents_dump_path)


def load_terms():
    return load_dump(terms_dump_path)


def dump_terms(terms):
    dump(terms, terms_dump_path)


def dump_documents(documents):
    dump(documents, documents_dump_path)


def dump_sparse(sparse):
    dump(sparse, sparse_dump_path)


def change_extension(filename, extension):
    file = filename.split('.')[0]
    return file + extension


def dump(serializable, path):
    with open(path, "wb") as file:
        pickle.dump(serializable, file)


def load_dump(path):
    serializable = None
    with open(path, "rb") as file:
        serializable = pickle.load(file)

    return serializable


# Todo this method should be called in an appropriate place!
def ensure_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find("corpora/words")
    except LookupError:
        nltk.download('words')
