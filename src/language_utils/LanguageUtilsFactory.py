from src.language_utils.ArabicUtils import ArabicUtils
from src.language_utils.HebrewUtils import HebrewUtils # This was likely missing
from src.language_utils.MalayUtils import MalayUtils
from src.language_utils.geez_utils import GeezUtils 

class LanguageUtilsFactory:
    @staticmethod
    def get_by_language(language: str):
        switch = {
            'he': HebrewUtils(),
            'ar': ArabicUtils(),
            'ms': MalayUtils(),
            'geez': GeezUtils(),
            'gez': GeezUtils(), # Support for the 'gez' key in params.py
        }
        return switch.get(language)