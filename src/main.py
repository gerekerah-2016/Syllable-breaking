import os
import sys

# Ensure the root directory is in the sys.path for imports to work in Colab
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.CorpusTokenizer import CorpusTokenizer
from src.SplinterTrainer import SplinterTrainer
from src.TextProcessorBaseline import TextProcessorBaseline
from src.TextProcessorWithEncoding import TextProcessorWithEncoding
from src.language_utils.LanguageUtilsFactory import LanguageUtilsFactory
from src.logger import initialize_logger, get_logger
from src.params import experiments, get_run_params, set_run_params
from src.save_dataset_as_text_file import save_corpus_as_text_file
from src.static_checks import run_static_checks, run_types_length_distribution
from src.train_tokenizer import train_tokenizer
from src.utils.path_utils import create_experiment_dirs, get_tokenizer_path, get_corpus_path, get_tokenized_corpus_path
from src.utils.utils import add_static_result_to_file, get_corpus_name

def run():
    language_utils = LanguageUtilsFactory.get_by_language(get_run_params("LANGUAGE"))
    train_dataset_path = get_run_params("SPLINTER_TRAINING_CORPUS_PATH")
    train_dataset_name = get_run_params("SPLINTER_TRAINING_CORPUS_NAME")
    letters_subset = get_run_params("SPLINTER_LETTERS_SUBSET")

    if get_run_params("SAVE_CORPORA_INTO_FILE"):
        if get_run_params("IS_ENCODED"):
            splinter_trainer = SplinterTrainer(language_utils)
            
            # FIXED: Capture all THREE values returned by train()
            reductions_map, new_unicode_chars_map, inverted_map = splinter_trainer.train(
                train_dataset_path, train_dataset_name, letters_subset
            )

            # FIXED: Pass all FOUR arguments to the constructor
            text_processor = TextProcessorWithEncoding(
                language_utils, 
                reductions_map, 
                new_unicode_chars_map, 
                inverted_map
            )
        else:
            text_processor = TextProcessorBaseline(language_utils)

        # Save corpora as text files
        save_corpus_as_text_file(text_processor, train_dataset_path, train_dataset_name)
        language_utils.save_additional_corpora_for_evaluation(text_processor)

    # Tokenizer Training
    if get_run_params("TRAIN_TOKENIZERS"):
        tokenizer_corpus_path = get_corpus_path(get_corpus_name(train_dataset_path, train_dataset_name))
        for t_type in get_run_params("TOKENIZERS_TYPES"):
            for v_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
                t_path = get_tokenizer_path(tokenizer_type=t_type, vocab_size=v_size)
                train_tokenizer(tokenizer_type=t_type, vocab_size=v_size, input_path=tokenizer_corpus_path, output_path=t_path)

    # Tokenization and Static Checks
    for t_type in get_run_params("TOKENIZERS_TYPES"):
        for v_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
            if get_run_params("TOKENIZE_CORPORA"):
                t_path = get_tokenizer_path(tokenizer_type=t_type, vocab_size=v_size)
                for c_name in get_run_params("STATIC_CHECKS_CORPORA"):
                    c_path = get_corpus_path(c_name)
                    token_path = get_tokenized_corpus_path(c_name, t_type, v_size)
                    CorpusTokenizer().tokenize_corpus_into_file(t_path, c_path, token_path)

            if get_run_params("RUN_STATIC_CHECKS"):
                dist = run_types_length_distribution(t_type, v_size)
                add_static_result_to_file(dist)
                for c_name in get_run_params("STATIC_CHECKS_CORPORA"):
                    c_path = get_corpus_path(c_name)
                    token_path = get_tokenized_corpus_path(c_name, t_type, v_size)
                    t_path = get_tokenizer_path(tokenizer_type=t_type, vocab_size=v_size)
                    res = run_static_checks(c_path, token_path, t_path, v_size)
                    add_static_result_to_file(res)

if __name__ == '__main__':
    slurm_id = os.getenv('SLURM_ARRAY_TASK_ID')
    task_id = (int(slurm_id) - 1) if slurm_id else 0
    experiment = experiments[task_id]
    set_run_params(experiment)
    create_experiment_dirs()
    initialize_logger()
    get_logger().info(f'Experiment {task_id + 1} started')
    try:
        run()
    except Exception as e:
        get_logger().exception(f'Experiment {task_id + 1} failed with error: \n{e}')
        exit(1)
    get_logger().info(f'Experiment {task_id + 1} finished')