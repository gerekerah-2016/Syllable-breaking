"""
Ge'ez Syllable Breaking package.
"""

from .syllable_breaker import GeEzSyllableBreaker
from .language_adapter import GeEzLanguageAdapter
from .dataset_loader import GeEzDatasetLoader

__version__ = "1.0.0"
__all__ = ["GeEzSyllableBreaker", "GeEzLanguageAdapter", "GeEzDatasetLoader"]