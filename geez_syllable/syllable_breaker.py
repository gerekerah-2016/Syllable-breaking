"""
Syllable Breaking implementation for Ge'ez/Ethiopic script.
Converts Ge'ez syllables to Virtual Abjad for SPLINTER processing.
"""

import os
import sys
import unicodedata

class GeEzSyllableBreaker:
    """
    Converts Ge'ez abugida syllables to consonant-vowel sequences.
    Solves syllabic sparsity by creating a "Virtual Abjad".
    """
    
    # Ethiopic Unicode range
    GEEZ_RANGE_START = 0x1200
    GEEZ_RANGE_END = 0x137F
    
    # Vowel markers (Private Use Area U+E000-U+E007)
    VOWEL_MARKERS = {
        0: '\uE000',  # Base form (no vowel)
        1: '\uE001',  # Vowel a
        2: '\uE002',  # Vowel u
        3: '\uE003',  # Vowel i
        4: '\uE004',  # Vowel aa
        5: '\uE005',  # Vowel ee
        6: '\uE006',  # Vowel e
        7: '\uE007',  # Vowel o
    }
    
    def __init__(self):
        """Initialize syllable breaker."""
        # Generate all Ge'ez characters
        self.all_geez_chars = [
            chr(i) for i in range(self.GEEZ_RANGE_START, self.GEEZ_RANGE_END + 1)
        ]
        
        # Generate base consonants (every 8th character starting from 0x1200)
        self.base_consonants = [
            chr(i) for i in range(
                self.GEEZ_RANGE_START, 
                self.GEEZ_RANGE_END + 1, 
                8
            )
        ]
        
        # Reverse mapping for restoration
        self.marker_to_vowel = {v: k for k, v in self.VOWEL_MARKERS.items()}
    
    def is_geez_character(self, char: str) -> bool:
        """Check if character is in Ethiopic range."""
        if len(char) != 1:
            return False
        code = ord(char)
        return self.GEEZ_RANGE_START <= code <= self.GEEZ_RANGE_END
    
    def break_syllable(self, syllable: str) -> tuple:
        """
        Break a Ge'ez syllable into (base_consonant, vowel_marker).
        
        Args:
            syllable: Single Ge'ez character (like 'ባ')
            
        Returns:
            Tuple of (base_consonant, vowel_marker)
        """
        if not self.is_geez_character(syllable):
            return (syllable, '')
        
        code = ord(syllable)
        vowel_order = code % 8  # Get vowel order (0-7)
        base_code = code - vowel_order  # Get base consonant code
        
        base_char = chr(base_code)
        vowel_marker = self.VOWEL_MARKERS.get(vowel_order, '')
        
        return (base_char, vowel_marker)
    
    def break_word(self, word: str) -> str:
        """
        Convert Ge'ez word to Virtual Abjad form.
        
        Example:
            "ባሱማ" → "በሰመ" + "⠁⠥⠁" (vowel markers)
        """
        result = []
        vowel_tags = []
        
        for char in word:
            if self.is_geez_character(char):
                base, vowel = self.break_syllable(char)
                result.append(base)
                if vowel:
                    vowel_tags.append(vowel)
            else:
                result.append(char)
        
        # Combine base consonants with vowel tags
        return ''.join(result) + ''.join(vowel_tags)
    
    def restore_word(self, broken_word: str) -> str:
        """
        Restore original Ge'ez word from Virtual Abjad.
        
        Example:
            "በሰመ" + "⠁⠥⠁" → "ባሱማ"
        """
        # Separate base consonants from vowel markers
        consonants = []
        vowels = []
        
        for char in broken_word:
            if char in self.marker_to_vowel:
                vowels.append(char)
            else:
                consonants.append(char)
        
        # Reconstruct syllables
        result = []
        vowel_index = 0
        
        for consonant in consonants:
            if (self.is_geez_character(consonant) and 
                vowel_index < len(vowels)):
                # Add vowel to consonant
                vowel_marker = vowels[vowel_index]
                vowel_order = self.marker_to_vowel[vowel_marker]
                original_code = ord(consonant) + vowel_order
                result.append(chr(original_code))
                vowel_index += 1
            else:
                result.append(consonant)
        
        return ''.join(result)
    
    def get_alphabet(self):
        """Get all Ge'ez characters for SPLINTER."""
        return self.all_geez_chars
    
    def preprocess_text_file(self, input_path: str, output_path: str):
        """
        Preprocess a text file: convert all Ge'ez words to Virtual Abjad.
        
        Args:
            input_path: Path to original Ge'ez text file
            output_path: Path to save processed text
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        processed_lines = []
        for line in lines:
            words = line.strip().split()
            processed_words = [self.break_word(word) for word in words]
            processed_lines.append(' '.join(processed_words))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
        
        print(f"Processed {len(lines)} lines from {input_path}")
        print(f"Saved to {output_path}")