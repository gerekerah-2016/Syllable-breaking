from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface

class GeezUtils(LanguageUtilsInterface):
    GEEZ_UNICODE_BLOCK = (0x1200, 0x137F)
    
    VOWEL_MARKERS = {
        0: "\uE001", 1: "\uE002", 2: "\uE003", 3: "\uE004",
        4: "\uE005", 5: "\uE006", 6: "\uE007"
    }
    MARKER_TO_ORDER = {v: k for k, v in VOWEL_MARKERS.items()}

    def __init__(self):
        self.alphabet = [chr(i) for i in range(self.GEEZ_UNICODE_BLOCK[0], self.GEEZ_UNICODE_BLOCK[1] + 1)]

    def remove_diacritics(self, text: str) -> str:
        """Ge'ez syllables are inherent; returning text unchanged."""
        return text

    def is_letter_in_language(self, letter: str):
        cp = ord(letter)
        is_geez = self.GEEZ_UNICODE_BLOCK[0] <= cp <= self.GEEZ_UNICODE_BLOCK[1]
        return len(letter) == 1 and (is_geez or letter in self.MARKER_TO_ORDER)

    def get_language_alphabet(self) -> [str]:
        return self.alphabet

    def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
        for c in word:
            if self.is_letter_in_language(c): 
                continue
            cp = ord(c)
            if 0x1360 <= cp <= 0x1368 or c.isspace(): 
                continue
            return True
        return False

    def replace_final_letters(self, word: str):
        """
        Step 1 & 3: Custom normalization (syllable breaking)
        Converts syllables into base consonant + PUA vowel tag.
        Example: 'ባሱማ' -> 'በ\uE001ሰ\uE002መ\uE003'
        """
        decomposed = ""
        for char in word:
            cp = ord(char)
            if self.GEEZ_UNICODE_BLOCK[0] <= cp <= self.GEEZ_UNICODE_BLOCK[1]:
                order = cp % 8 
                base_char = chr(cp - order)
                vowel_tag = self.VOWEL_MARKERS.get(order, "")
                decomposed += base_char + vowel_tag
            else:
                decomposed += char
        return decomposed

    def save_additional_corpora_for_evaluation(self, text_processor) -> None:
        pass
