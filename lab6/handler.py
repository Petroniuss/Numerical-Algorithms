import sys
import os
import re  
from nltk.stem.porter import PorterStemmer
from bs4              import BeautifulSoup

black_list = ['script', 'style']


def words_from_html(document):
    content = None
    with open(document, 'r', encoding = 'UTF-8') as file:
        html_doc_raw = file.read()
        soup         = BeautifulSoup(html_doc_raw, 'html.parser')
        # Remove redundant tags
        for tag in soup(black_list):
            tag.extract()

        content = soup.get_text()
    
    content = re.sub(r'\W', ' ', content)
    words = content.split()

    return words


# Returns set of stems extraced from given document.
def stems_from_html(document):
    stemmer = PorterStemmer()
    stems = set()

    words = words_from_html(document)
    for word in words:
        if word.isalpha():
            stem = stemmer.stem(word)
            if stem not in stems:
                stems.add(stem.lower())

    return stems

# Returns set of words which form dictionary
# This is very slow :/

# As this is very slow we could probably download english dictionary
# and this could be our Bag of Words.
# But still the problem remains we need to traverse through all of these files
# to calculate frequency ... 
def dictionary_union():
    dictionary = set()
    for document in os.listdir('docs'):
        dictionary |= stems_from_html('docs' + document)
        break

    print(dictionary)

    return remove_stop_words(dictionary)

# create dictionary (stems => freq) and iterate over each word in document - easy.
# How to do all these operations for 50k documents!!?!!?
def calculate_freq_vector(dictionary):
    pass


# Modifies given set of stems leaving those which happen not to be stop words 
def remove_stop_words(stems):
    stop_words_set = set()
    with open("resources/english-stop-word.txt", "r", encoding = 'UTF-8') as file:
        for word in file.read().split():
            stop_words_set.add(word)
    
    return stems.difference(stop_words_set)

if __name__ == "__main__":
    print('Starting this super extra important task...')
    
    # words = stems_from_html(sys.argv[1])
    # print(len(remove_stop_words(words)))
    print(words_from_html(sys.argv[1]))

    print('Done...')