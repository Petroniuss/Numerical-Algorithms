import nltk
import sys
import os
import re
import string
from nltk.corpus import stopwords
from resources_manager import data_dir
from collections import Counter
from Document import Document
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer


# In case we'd like to leave only english words
english_words = set(nltk.corpus.words.words())
alphabet = set(list(string.ascii_letters)) | set(list(string.digits))


def is_in_alphabet(word):
    for char in word:
        if char not in alphabet:
            return False

    return True

# Retrives words from given html file.
# Words containing non-english characters are removed.
# Returned words are all lowercase.
# I set it to true - since either way we'll have way more than 10k words.


def words_from_html(path, remove_non_english_words=True):
    content = None
    with open(path, 'r', encoding='UTF-8') as file:
        html_doc_raw = file.read()
        soup = BeautifulSoup(html_doc_raw, 'html.parser')
        # Remove redundant tags
        for tag in soup(['script', 'style']):
            tag.extract()

        content = soup.get_text(separator=' ')

    content = re.sub(r'\W', ' ', content)
    words = content.split()
    words = [word.lower()
             for word in words if word.isalpha() and is_in_alphabet(word)]
    if remove_non_english_words:
        words = [word for word in words if word in english_words]

    return words

# Returns set of stems extacted from given words.


def extract_stems(words):
    stemmer = PorterStemmer()
    stems = [stemmer.stem(word) for word in words]

    return stems

# Returns set of words without english stop words.


def remove_stop_words(words):
    stop_words_set = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words_set]

    return words

# Processes given path to html file, returns bag of words.


def process_html(path):
    words = words_from_html(path)
    words = remove_stop_words(words)
    stems = extract_stems(words)

    bag_of_words = Counter(stems)

    return bag_of_words


def preprocess_all(directory):
    """ It returns list of documents, and map of words: word => (index, frequency) """
    files = os.listdir(directory)
    files_total = len(files)
    documents = []
    words_union = Counter()

    print('Preprocessor -- processing files...')
    for i, filename in enumerate(files):
        if not filename.endswith('.html'):
            raise Exception('Supplied directory contains non-html files.')

        path = os.path.join(directory, filename)
        words_dict = process_html(path)

        document = Document(filename, words_dict)
        documents.append(document)

        words_counter = Counter()
        for word in words_dict.keys():
            words_counter[word] = 1

        words_union += words_counter

        print(f'Progress: {i}/{files_total}', end='\r')

    for i, (word, freq) in enumerate(words_union.items()):
        words_union[word] = (i, freq)

    print(f'Bag of words contains {len(words_union)}.')
    print('Preprocessor -- done.')

    return documents, words_union
