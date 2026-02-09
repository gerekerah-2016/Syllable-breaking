import re
from src.TextProcessorInterface import TextProcessorInterface
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.utils.utils import get_permutation

class TextProcessorWithEncoding(TextProcessorInterface):

    def __init__(self, language_utils: LanguageUtilsInterface, reductions_map, new_unicode_chars_map, new_unicode_chars_inverted_map):
        super().__init__(language_utils)
        self.reductions_map = reductions_map
        self.new_unicode_chars_map = new_unicode_chars_map
        self.new_unicode_chars_inverted_map = new_unicode_chars_inverted_map
        self.word_reductions_cache = dict()

    def process(self, text):
        if text is None:
            return ''

        clean_text = self.language_utils.remove_diacritics(text)
        sentences = re.split(r'[.\n]', clean_text)
        encoded_sentences_list = list()
        
        for sentence in sentences:
            encoded_words_list = list()
            """words = re.split(r'\s|-|,|:|"|\(|\)', sentence)"""
            # This splits on whitespace AND common Ge'ez/Latin punctuation
            words = re.split(r'[\s፡።፣፤፤፥፦፧፨\-,:;()\"\'?!]+', sentence)
            words = [w for w in words if w] # Filter out empty strings

            for word in words:
                if not word:
                    continue
                
                original_word = word
                if original_word in self.word_reductions_cache:
                    encoded_word = self.word_reductions_cache[original_word]
                else:
                    word = self.language_utils.replace_final_letters(word)
                    if self.language_utils.is_word_contains_letters_from_other_languages(word):
                        encoded_word = self.get_reduction_for_word_with_letters_from_other_languages(word)
                    else:
                        word_reductions = self.get_word_reductions(word)
                        # Handle both splinter keys and raw characters safely using .get()
                        encoded_chars = [self.new_unicode_chars_map.get(red, red) for red in word_reductions]
                        encoded_word = ''.join(encoded_chars)
                
                encoded_words_list.append(encoded_word)
                self.word_reductions_cache[original_word] = encoded_word
            
            encoded_sentence = " ".join(encoded_words_list)
            if encoded_sentence:
                encoded_sentences_list.append(encoded_sentence)
                
        return "\n".join(encoded_sentences_list)

    def get_word_reductions(self, word):
        reduced_word = word
        reductions = []
        while len(reduced_word) > 3:
            if len(reduced_word) not in self.reductions_map:
                reductions.extend(self.get_single_chars_reductions(reduced_word))
                break
            reduction = self.get_reduction(reduced_word, 3, 3)
            if reduction is not None:
                position = int(reduction.split(':')[0])
                reductions.append(reduction)
                reduced_word = get_permutation(reduced_word, position, len(reduced_word))
            else:
                reductions.extend(self.get_single_chars_reductions(reduced_word))
                break

        if len(reduced_word) < 4:
            reductions.extend(self.get_single_chars_reductions(reduced_word))

        reductions.reverse()
        return reductions

    def get_reduction_for_word_with_letters_from_other_languages(self, word):
        # Fix: Don't look up individual characters in the map.
        # Just return the word as-is (already syllable-broken)
        return word

    def rebuild_reduced_word(self, encoded_text):
        if isinstance(encoded_text, list):
            decoded_parts = []
            for token in encoded_text:
                if token in self.new_unicode_chars_inverted_map:
                    splinter_key = self.new_unicode_chars_inverted_map[token]
                    decoded_parts.append(splinter_key.split(':')[-1])
                elif ':' in token and token.split(':')[0].isdigit():
                    decoded_parts.append(token.split(':')[-1])
                else:
                    decoded_parts.append(token)
            return "".join(decoded_parts)
        return encoded_text

    def get_reduction(self, word, depth, width):
        curr_step_reductions = [{"word": word, "reduction": None, "root_reduction": None, "score": 1}]
        word_length = len(word)
        i = 0
        while i < depth and len(curr_step_reductions) > 0 and word_length > 3:
            next_step_reductions = list()
            for reduction in curr_step_reductions:
                possible_reductions = self.get_most_frequent_reduction_keys(
                    reduction["word"],
                    reduction["root_reduction"],
                    reduction["score"],
                    width,
                    word_length
                )
                next_step_reductions += possible_reductions
            curr_step_reductions = list(next_step_reductions)
            i += 1
            word_length -= 1

        max_score_reduction = None
        if len(curr_step_reductions) > 0:
            max_score_reduction = max(curr_step_reductions, key=lambda x: x["score"])["root_reduction"]
        return max_score_reduction

    def get_most_frequent_reduction_keys(self, word, root_reduction, parent_score, number_of_reductions, word_length):
        if len(word) not in self.reductions_map:
            return list()

        possible_reductions = list()
        for reduction, score in self.reductions_map[len(word)].items():
            parts = reduction.split(':')
            if len(parts) < 2: continue
            position, letter = int(parts[0]), parts[1]
            if position < len(word) and word[position] == letter:
                permutation = get_permutation(word, position, word_length)
                possible_reductions.append({
                    "word": permutation,
                    "reduction": reduction,
                    "root_reduction": root_reduction if root_reduction is not None else reduction,
                    "score": parent_score * score
                })
                if len(possible_reductions) >= number_of_reductions:
                    break
        return possible_reductions

    @staticmethod
    def get_single_chars_reductions(reduced_word):
        reductions = []
        # Ensure we are returning a list of characters
        for char in reduced_word[::-1]:
            reductions.append(char)
        return reductions # Removed the trailing "/content"