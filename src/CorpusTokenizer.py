"""import os

import sentencepiece as spm
from tqdm import tqdm

from src.logger import get_logger


class CorpusTokenizer:

    @staticmethod
    def tokenize_corpus_into_file(tokenizer_path, corpus_path, output_path):
        if os.path.exists(output_path):
            os.remove(output_path)

        get_logger().info(f'start tokenizing corpus {output_path}')
        sp_tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')
        with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
            with open(output_path, 'a', encoding='utf-8') as output_file:
                for article_text in tqdm(corpus_file):
                    if article_text is not None:
                        encoded_article = " ".join([str(token_id) for token_id in sp_tokenizer.encode(article_text)])
                        output_file.write(encoded_article + '\n')
        get_logger().info(f'finished tokenizing corpus {output_path}')
import os

import sentencepiece as spm
from tqdm import tqdm

from src.logger import get_logger


class CorpusTokenizer:

    @staticmethod
    def tokenize_corpus_into_file(tokenizer_path, corpus_path, output_path):
        if os.path.exists(output_path):
            os.remove(output_path)

        get_logger().info(f'start tokenizing corpus {output_path}')
        sp_tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')
        with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
            with open(output_path, 'a', encoding='utf-8') as output_file:
                for article_text in tqdm(corpus_file):
                    if article_text is not None:
                        encoded_article = " ".join([str(token_id) for token_id in sp_tokenizer.encode(article_text)])
                        output_file.write(encoded_article + '\n')
        get_logger().info(f'finished tokenizing corpus {output_path}')
import os

import sentencepiece as spm
from tqdm import tqdm

from src.logger import get_logger


class CorpusTokenizer:

    @staticmethod
    def tokenize_corpus_into_file(tokenizer_path, corpus_path, output_path):
        if os.path.exists(output_path):
            os.remove(output_path)

        get_logger().info(f'start tokenizing corpus {output_path}')
        sp_tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')
        with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
            with open(output_path, 'a', encoding='utf-8') as output_file:
                for article_text in tqdm(corpus_file):
                    if article_text is not None:
                        encoded_article = " ".join([str(token_id) for token_id in sp_tokenizer.encode(article_text)])
                        output_file.write(encoded_article + '\n')
        get_logger().info(f'finished tokenizing corpus {output_path}')
import os

import sentencepiece as spm
from tqdm import tqdm

from src.logger import get_logger


class CorpusTokenizer:

    @staticmethod
    def tokenize_corpus_into_file(tokenizer_path, corpus_path, output_path):
        if os.path.exists(output_path):
            os.remove(output_path)

        get_logger().info(f'start tokenizing corpus {output_path}')
        sp_tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')
        with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
            with open(output_path, 'a', encoding='utf-8') as output_file:
                for article_text in tqdm(corpus_file):
                    if article_text is not None:
                        encoded_article = " ".join([str(token_id) for token_id in sp_tokenizer.encode(article_text)])
                        output_file.write(encoded_article + '\n')
        get_logger().info(f'finished tokenizing corpus {output_path}')
import os

import sentencepiece as spm
from tqdm import tqdm

from src.logger import get_logger


class CorpusTokenizer:

    @staticmethod
    def tokenize_corpus_into_file(tokenizer_path, corpus_path, output_path):
        if os.path.exists(output_path):
            os.remove(output_path)

        get_logger().info(f'start tokenizing corpus {output_path}')
        sp_tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')
        with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
            with open(output_path, 'a', encoding='utf-8') as output_file:
                for article_text in tqdm(corpus_file):
                    if article_text is not None:
                        encoded_article = " ".join([str(token_id) for token_id in sp_tokenizer.encode(article_text)])
                        output_file.write(encoded_article + '\n')
        get_logger().info(f'finished tokenizing corpus {output_path}')
"""
import os
import sentencepiece as spm
from tqdm import tqdm
from src.logger import get_logger

class CorpusTokenizer:

    @staticmethod
    def tokenize_corpus_into_file(tokenizer_path, corpus_path, output_path, language_utils=None):
        """
        Tokenizes the corpus. 
        NOTE: For Ge'ez, ensure 'language_utils' is passed so 'article_text' 
        can be decomposed into the Root + PUA format before encoding.
        """
        if os.path.exists(output_path):
            os.remove(output_path)

        get_logger().info(f'start tokenizing corpus {output_path}')
        sp_tokenizer = spm.SentencePieceProcessor(model_file=f'{tokenizer_path}.model')
        
        with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
            with open(output_path, 'a', encoding='utf-8') as output_file:
                for article_text in tqdm(corpus_file):
                    if article_text is not None:
                        # --- GE'EZ PRE-PROCESSING STEP ---
                        # If language_utils is provided, we must convert raw Ge'ez syllables
                        # into the decomposed format learned during the SplinterTrainer phase.
                        if language_utils:
                            article_text = language_utils.replace_final_letters(article_text)
                        
                        # The tokenizer now sees the 'splintered' text and maps it to IDs
                        encoded_article = " ".join([str(token_id) for token_id in sp_tokenizer.encode(article_text)])
                        output_file.write(encoded_article + '\n')
                        
        get_logger().info(f'finished tokenizing corpus {output_path}')