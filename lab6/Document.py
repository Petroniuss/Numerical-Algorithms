import os
from resources_manger import processed_data_dir, \
                             data_dir, \
                             load_dump, \
                             dump, \
                             change_extension \

class Document:
    def __init__(self, filename, words_dict): 
        self.filename = filename
        self.save_words_dict(words_dict)

    def link(self):
        return os.path.join(data_dir, self.filename)

    def words_dict(self):
        return load_dump(self.dictionary_dump_path())

    def save_words_dict(self, words_dict):
        dump(words_dict, self.dictionary_dump_path())

    def dictionary_dump_path(self):
        dump_filename = change_extension(self.filename, '-dict.pickle')
        dump_path     = os.path.join(processed_data_dir, dump_filename)

        return dump_path  