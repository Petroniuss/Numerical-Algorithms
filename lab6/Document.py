import os
from resources_manager import processed_data_dir, \
    data_dir, \
    change_extension
import resources_manager


class Document:
    def __init__(self, filename, words_dict):
        self.filename = filename
        self.save_words_dict(words_dict)

    def __str__(self):
        header = '===' * 3 + 'Document' + '===' * 3
        line = f'{self.filename}'.center(26)

        return header + '\n' + line + '\n'

    def __repr__(self):
        return str(self)

    def link(self):
        return 'https://en.wikipedia.org/wiki/' + self.filename.replace('.html', '')

    def words_dict(self):
        return resources_manager.load_dump(self.dictionary_dump_path())

    def save_words_dict(self, words_dict):
        resources_manager.dump(words_dict, self.dictionary_dump_path())

    def dictionary_dump_path(self):
        dump_filename = change_extension(self.filename, '-dict.pickle')
        dump_path = os.path.join(processed_data_dir, dump_filename)

        return dump_path
