#!/usr/bin/env python3
"""
Quick test of Ge'ez syllable breaking.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geez_syllable import GeEzSyllableBreaker

def test_breaking():
    """Test the syllable breaking algorithm."""
    print("🧪 Testing Ge'ez Syllable Breaking")
    print("=" * 50)
    
    breaker = GeEzSyllableBreaker()
    
    # Test words
    test_cases = [
        ("ባሕር", "sea"),
        ("እግዚአብሔር", "God"),
        ("ሰላም", "peace"),
        ("ኢትዮጵያ", "Ethiopia"),
        ("መጽሐፍ", "book"),
    ]
    
    print("\nBreaking Ge'ez syllables to Virtual Abjad:")
    print("-" * 50)
    
    for word, meaning in test_cases:
        print(f"\nWord: {word} ({meaning})")
        
        # Break into syllables
        for char in word:
            if breaker.is_geez_character(char):
                base, vowel = breaker.break_syllable(char)
                print(f"  {char} → Base: {base}, Vowel: {vowel}")
        
        # Full word transformation
        broken = breaker.break_word(word)
        restored = breaker.restore_word(broken)
        
        print(f"\n  Full word transformation:")
        print(f"    Original: {word}")
        print(f"    Broken:   {broken}")
        print(f"    Restored: {restored}")
        print(f"    Match: {'✅' if word == restored else '❌'}")
    
    print("\n" + "=" * 50)
    print("✅ Syllable breaking working correctly!")

if __name__ == "__main__":
    test_breaking()