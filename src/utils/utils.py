import json
import re
from collections import Counter, defaultdict
from src.utils.path_utils import get_splinter_dir, get_logs_dir

def get_words_dict_by_length(words_dict):
    """Groups words from the dictionary by their character length."""
    words_by_length = defaultdict(dict)
    for word, count in words_dict.items():
        words_by_length[len(word)][word] = count
    return dict(words_by_length)

def get_permutation(word, position, word_length):
    """Removes a character at a specific position to create a shorter word."""
    return word[:position] + word[position+1:]

def get_corpus_name(dataset_path, dataset_name):
    """Generates a clean name for the corpus files based on path/name."""
    name = dataset_name.replace('\\', '/').split('/')[-1]
    return name.replace('.txt', '').replace('.', '_')

def add_static_result_to_file(result):
    with open(f'{get_logs_dir()}/static_checks_results.json', 'a', encoding='utf-8') as file:
        file.write('\n')
        json.dump(result, file, indent='\t', ensure_ascii=False)

def get_reductions_map_from_file():
    with open(f'{get_splinter_dir()}/reductions_map.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    return {int(key): value for key, value in data.items()}

def get_new_unicode_chars_map_from_file():
    with open(f'{get_splinter_dir()}/new_unicode_chars.json', 'r', encoding='utf-8') as file:
        return json.load(file)

def get_new_unicode_chars_inverted_map_from_file():
    with open(f'{get_splinter_dir()}/new_unicode_chars_inverted.json', 'r', encoding='utf-8') as file:
        return json.load(file)

def get_letters_frequency(words_dict):
    letters_frequency = defaultdict(int)
    for word, count in words_dict.items():
        for char in word:
            letters_frequency[char] += count
    return dict(sorted(letters_frequency.items(), key=lambda item: item[1], reverse=True))

def decode_tokens_vocab_file(tokens_vocab_file):
    with open(f'{tokens_vocab_file}.vocab', 'r', encoding='utf-8') as file:
        encoded_tokens = file.read().splitlines()
    
    inverted_map = get_new_unicode_chars_inverted_map_from_file()
    decoded_tokens = []
    for line in encoded_tokens:
        parts = line.split('\t')
        token = parts[0]
        decoded_word = "".join([inverted_map.get(char, char) for char in token])
        decoded_tokens.append(f"{decoded_word}\t{parts[1] if len(parts) > 1 else ''}")

    with open(f'{tokens_vocab_file}_decoded.vocab', 'w', encoding='utf-8') as file:
        file.write("\n".join(decoded_tokens))