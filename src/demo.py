import os
import sentencepiece as spm

from src.TextProcessorForDemo import TextProcessorForDemo
from src.language_utils.LanguageUtilsFactory import LanguageUtilsFactory
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.logger import get_logger, initialize_logger
from src.params import set_run_params, get_dummy_experiment
from src.utils.path_utils import get_logs_dir, get_tokenizers_dir
from src.utils.utils import (get_reductions_map_from_file, get_new_unicode_chars_map_from_file,
                             get_new_unicode_chars_inverted_map_from_file)

def demo(text: str, language_utils: LanguageUtilsInterface) -> None:
    get_logger().info(f'\nOriginal text: \n{text}')
    text_processor = get_processor(language_utils)
    
    # Process converts raw Ge'ez -> Decomposed (Root + PUA) -> Splintered encoding
    non_encoded_text, encoded_text = text_processor.process(text)
    
    get_logger().info(f'\n\nNon-encoded splintered text: \n{non_encoded_text}')
    get_logger().info(f'\n\nEncoded splintered text: \n{encoded_text}')
    
    # undo_process converts Splintered encoding -> Decomposed -> Raw Ge'ez syllables
    decoded_text, original_text = text_processor.undo_process(encoded_text)
    
    get_logger().info(f'\n\nDecoded splintered text: \n{decoded_text}')
    get_logger().info(f'\n\nBack to original text: \n{original_text}')

def get_processor(language_utils: LanguageUtilsInterface) -> TextProcessorForDemo:
    reductions_map = get_reductions_map_from_file()
    new_unicode_chars_map = get_new_unicode_chars_map_from_file()
    new_unicode_chars_inverted_map = get_new_unicode_chars_inverted_map_from_file()
    text_processor = TextProcessorForDemo(language_utils, reductions_map, new_unicode_chars_map, new_unicode_chars_inverted_map)
    return text_processor

def tokenize_example(text: str, tokenizer_type: str, vocab_size: int) -> None:
    tokenizer_path = f'{get_tokenizers_dir()}/{tokenizer_type}_{vocab_size}.model'
    sp_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    encoded_text = " ".join([str(token_id) for token_id in sp_tokenizer.encode(text)])
    get_logger().info(f'tokenized example: {encoded_text}')

if __name__ == '__main__':
    # --- PREVIOUS CODE (Commented Out) ---
    # experiment = get_dummy_experiment('2025-01-21-hebrew-letters_subset')
    # demo("נגרר", LanguageUtilsFactory.get_by_language("he"))

    # --- GE'EZ DEMO CODE ---
    # Ensure this name matches your current Ge'ez experiment folder in /logs
    experiment = get_dummy_experiment('2025-01-21-geez-all_letters')
    set_run_params(experiment)
    
    os.makedirs(get_logs_dir(), exist_ok=True)
    initialize_logger()
    get_logger().info(f'Logs will be saved here: {get_logs_dir()}')

    # Example Ge'ez word for testing (e.g., "መሠረት" - foundation)
    geez_test_word = "መሠረት" 
    
    # We use the factory to get your GeezUtils implementation
    geez_utils = LanguageUtilsFactory.get_by_language("gez")
    
    # Run the demo
    demo(geez_test_word, geez_utils)