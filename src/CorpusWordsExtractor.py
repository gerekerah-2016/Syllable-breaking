import json
import os
import re
from collections import defaultdict

class CorpusWordsExtractor:
    def __init__(self, language_utils, dataset_name=None):
        self.language_utils = language_utils
        self.dataset_name = dataset_name

    def convert_corpus_to_words_dict_file(self, corpus, output_filename):
        words = self.get_words_from_corpus(corpus) 
    
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False, indent=4)
        return words    

    def get_words_from_corpus(self, corpus):
        articles_text = [item['text'] for item in corpus]
        return self.extract_words_with_frequencies(articles_text)

    def extract_words_with_frequencies(self, articles_text):    
        if self.language_utils is None:
            raise ValueError("CorpusWordsExtractor initialized with NoneType language_utils. Check SplinterTrainer init.")

        words = defaultdict(int)
        for article_text in articles_text:
            if article_text is None: 
                continue
            
            article_text = self.language_utils.remove_diacritics(article_text)
            article_words = re.split(r'\.|\s|\n|-|,|:|"|\(|\)', article_text)

            for word in article_words:
                if word:
                    processed_word = self.language_utils.replace_final_letters(word)
                    words[processed_word] += 1
                    
        return dict(words)