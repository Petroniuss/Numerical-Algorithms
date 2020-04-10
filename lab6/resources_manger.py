import pickle
import nltk
import os

resources_path = os.path.join(os.getcwd(), 'resources')

data_dir       = os.path.join(resources_path, 'data')
processed_dir  = os.path.join(resources_path, 'processed')

processed_data_dir = os.path.join(processed_dir, 'data')


### -------------------------------------|
### Contains utilties:                   |
###     Saving & loading proccessed data`|
### -------------------------------------|

# Todo calculate matrices and save them to files!

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