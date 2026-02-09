import json
import os
import numpy as np
from datasets import load_dataset
from src.CorpusWordsExtractor import CorpusWordsExtractor
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.logger import get_logger
from src.utils.path_utils import get_logs_dir, get_raw_data_dir, get_splinter_dir, get_words_dict_dir
from src.utils.utils import get_words_dict_by_length, get_permutation, get_corpus_name

class SplinterTrainer:
    def __init__(self, language_utils: LanguageUtilsInterface):
        if language_utils is None:
            raise ValueError("SplinterTrainer initialized with NoneType language_utils.")
        self.language_utils = language_utils

    def train(self, dataset_path: str, dataset_name: str, letters_for_reductions: [str] = None):
        words_dict = self.get_word_dict(dataset_path, dataset_name)
        
        #  Filter and pre-process words (canonicalization for Ge'ez)
        pre_process_words_dict = self.pre_process_words(words_dict)
        
        if not pre_process_words_dict:
            raise ValueError("The word dictionary is empty after pre-processing. Check your frequency threshold and language filters.")
        words_dict_by_length = get_words_dict_by_length(pre_process_words_dict)
        max_length = sorted(words_dict_by_length.keys(), reverse=True)[0]
    
        get_logger().info(f"Start first iteration of reductions:")
        reductions = self.initialize_reductions_dict(max_length)
        for current_length in range(max_length, 1, -1):
            if current_length not in words_dict_by_length:
                continue
            
            current_length_words = words_dict_by_length[current_length]
            previous_length_words = words_dict_by_length.get(current_length - 1, {})
            
            for word, count in current_length_words.items():
                best_reduction = self.get_reduction(word, reductions, previous_length_words, max_number_of_candidates=10)
                if best_reduction:
                    reduction_key = best_reduction["reduction"]
                    reductions[current_length][reduction_key] = reductions[current_length].get(reduction_key, 0) + count

        # 4. Map top reductions to Private Use Area (PUA) Unicode for tokenization
        new_unicode_chars_map = {}
        new_unicode_chars_inverted_map = {}
        next_unicode = 0xE000  # Start of PUA
        
        final_reductions = {}
        for length, length_reductions in reductions.items():
            if not length_reductions: 
                continue
            sorted_reds = dict(sorted(length_reductions.items(), key=lambda item: item[1], reverse=True)[:8000])
            final_reductions[length] = sorted_reds
            
            for red_key in sorted_reds.keys():
                if red_key not in new_unicode_chars_map:
                    char = chr(next_unicode)
                    new_unicode_chars_map[red_key] = char
                    new_unicode_chars_inverted_map[char] = red_key
                    next_unicode += 1
        self.save_result_file('reductions_map', final_reductions)
        self.save_result_file('new_unicode_chars', new_unicode_chars_map)
        self.save_result_file('new_unicode_chars_inverted', new_unicode_chars_inverted_map)
        
        return final_reductions, new_unicode_chars_map, new_unicode_chars_inverted_map

    def get_word_dict(self, dataset_path, dataset_name):
        # Check if we are using the Hugging Face dataset
        if dataset_path == "amanuelbyte/Amharic_dataset":
            print(f"Loading dataset from Hugging Face: {dataset_path}")
            dataset = load_dataset(dataset_path, streaming=True)
        
            # We use the 'train' split and the 'text' column based on your log
            # To save memory/time, you can use a subset: dataset['train'].select(range(100000))
            data_iterator = dataset['train']['text'] 
        else:
            # Fallback for local files if needed
            dataset = load_dataset('text', data_files=dataset_path)
            data_iterator = dataset['train']['text']

        # Your existing extraction logic starts here:
        corpus_word_extractor = CorpusWordsExtractor(self.language_utils, dataset_name)
        words_dict = corpus_word_extractor.extract_words_with_frequencies(data_iterator)
    
        return words_dict   
    def pre_process_words(self, words_dict):
        """Filters noise and applies language-specific canonicalization (e.g., Ge'ez roots)."""
        processed_dict = {}
        for word, count in words_dict.items():
           
            # replace_final_letters converts syllables to (Consonant + PUA Vowel) 
            # for Ge'ez if not already done in the Extractor.
            clean_word = self.language_utils.replace_final_letters(word)
            
            if not self.language_utils.is_word_contains_letters_from_other_languages(clean_word):
                processed_dict[clean_word] = processed_dict.get(clean_word, 0) + count
        
        # Filter: keep words with length > 1 and frequency > 1
        return {word: count for word, count in processed_dict.items() if len(word) > 1 and count > 1}

    def initialize_reductions_dict(self, max_length):
        return {i: {} for i in range(2, max_length + 1)}

    def get_reduction(self, word, reductions, previous_length_words, max_number_of_candidates):
        word_length = len(word)
        possible_reductions = []
        
        for i in range(word_length):
            permutation = get_permutation(word, i, word_length)
            if permutation in previous_length_words:
                reduction_key = f"{i}:{word[i]}"
                score = previous_length_words[permutation]
                possible_reductions.append({"reduction": reduction_key, "score": score})
        
        if not possible_reductions:
            return None
            
        possible_reductions.sort(key=lambda x: x["score"], reverse=True)
        return possible_reductions[0]

    def save_result_file(self, file_name, data):
        output_dir = get_splinter_dir()
        os.makedirs(output_dir, exist_ok=True)
            
        target_path = f'{output_dir}/{file_name}.json'
        with open(target_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent='\t', ensure_ascii=False)
        log_path = f'{get_logs_dir()}/{file_name}.json'
        os.makedirs(get_logs_dir(), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent='\t', ensure_ascii=False)