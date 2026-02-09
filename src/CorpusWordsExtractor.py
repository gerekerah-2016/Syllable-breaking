import json
import os
import re

from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.logger import get_logger
from src.utils.path_utils import get_words_dict_dir


class CorpusWordsExtractor:

    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils

    def convert_corpus_to_words_dict_file(self, corpus, output_filename):
        # CHANGE: Instead of passing corpus["text"], we pass the iterable corpus object
        words = self.get_words_from_corpus(corpus)
        if not os.path.exists(get_words_dict_dir()):
            os.makedirs(get_words_dict_dir())
        with open(f'{get_words_dict_dir()}/{output_filename}.json', 'w', encoding='utf-8') as file:
            json.dump(words, file, indent='\t', ensure_ascii=False)

    def get_words_from_corpus(self, corpus_iterable):
        words = {}
        # CHANGE: Iterate through the dataset rows to support streaming (IterableDataset)
        for index, row in enumerate(corpus_iterable):
            article_text = row["text"] # Access the text column from the row
            article_text = self.language_utils.remove_diacritics(article_text)
            article_words = re.split(r'\.|\s|\n|-|,|:|"|\(|\)', article_text)

            for word in article_words:
                if not word: continue
                if word not in words:
                    words[word] = 0
                words[word] += 1

            if (index + 1) % 10000 == 0:
                get_logger().info(f'Finished extracting words from {index + 1} articles')
        
        get_logger().info(f'Finished extracting words from total articles.')
        return words