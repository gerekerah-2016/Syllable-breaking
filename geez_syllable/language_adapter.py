"""
Adapter to make Syllable Breaker work with SPLINTER's LanguageUtilsInterface.
This allows us to use existing SPLINTER code without modifying it.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
except ImportError:
    # Create a dummy interface if SPLINTER isn't available
    class LanguageUtilsInterface:
        def remove_diacritics(self, text: str) -> str:
            pass
        def is_letter_in_language(self, char: str) -> bool:
            pass
        def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
            pass
        def get_language_alphabet(self):
            pass
        def replace_final_letters(self, text: str) -> str:
            pass
        def save_additional_corpora_for_evaluation(self, text_processor) -> None:
            pass

from .syllable_breaker import GeEzSyllableBreaker

class GeEzLanguageAdapter(LanguageUtilsInterface):
    """
    Adapter that makes GeEzSyllableBreaker compatible with SPLINTER.
    Implements all required methods of LanguageUtilsInterface.
    """
    
    def __init__(self):
        """Initialize adapter with syllable breaker."""
        self.breaker = GeEzSyllableBreaker()
    
    def remove_diacritics(self, text: str) -> str:
        """
        Ge'ez doesn't have separate diacritics.
        Return normalized form.
        """
        return unicodedata.normalize('NFC', text)
    
    def is_letter_in_language(self, char: str) -> bool:
        """Check if character is Ge'ez."""
        return self.breaker.is_geez_character(char)
    
    def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
        """
        Check if word contains non-Geez characters.
        Only checks the base part (ignores vowel markers).
        """
        for char in word:
            if (char not in self.breaker.marker_to_vowel and 
                not self.breaker.is_geez_character(char) and
                char not in ' \t\n\r'):
                return True
        return False
    
    def get_language_alphabet(self):
        """Return Ge'ez alphabet for SPLINTER."""
        return self.breaker.get_alphabet()
    
    def replace_final_letters(self, text: str) -> str:
        """
        MAIN INTEGRATION POINT: Called by SPLINTER.
        Applies syllable breaking to transform Ge'ez to Virtual Abjad.
        """
        return self.breaker.break_word(text)
    
    def save_additional_corpora_for_evaluation(self, text_processor) -> None:
        """No additional corpora needed for Ge'ez."""
        pass
    
    @staticmethod
    def replace_last_letter(text, replacement):
        """Utility method required by interface."""
        return text[:-1] + replacement