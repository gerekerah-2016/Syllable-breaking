import os
# Change this path to a folder on your D: drive where you want the data stored
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = 'D:/huggingface_cache/datasets'
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 2. NOW import your project files
from src.CorpusTokenizer import CorpusTokenizer
from src.SplinterTrainer import SplinterTrainer
# ... (rest of your imports)

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
import sys

# This forces Python to look at your current folder first
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run():
    # 1. Initialize Ge'ez utilities from your custom class
    language_utils = LanguageUtilsFactory.get_by_language(get_run_params("LANGUAGE"))
    train_dataset_path = get_run_params("SPLINTER_TRAINING_CORPUS_PATH")
    train_dataset_name = get_run_params("SPLINTER_TRAINING_CORPUS_NAME")
    letters_subset = get_run_params("SPLINTER_LETTERS_SUBSET")

    # 2. Handle Splintering vs. Baseline
    if get_run_params("SAVE_CORPORA_INTO_FILE"):
        # This branch performs the Ge'ez syllable decomposition (Breaking)
        
        
        if get_run_params("IS_ENCODED"):
            get_logger().info("Starting Splinter Training for Ge'ez...")    
            splinter_trainer = SplinterTrainer(language_utils)
            
            # ✅ Change 1: Capture the third return value (inverted map)
            reductions_map, new_unicode_chars_map, new_unicode_chars_inverted_map = splinter_trainer.train(
                train_dataset_path, train_dataset_name, letters_subset
            )
            
            # ✅ Change 2: Pass all 4 required arguments to the processor
            text_processor = TextProcessorWithEncoding(
                language_utils, 
                reductions_map, 
                new_unicode_chars_map, 
                new_unicode_chars_inverted_map
            )    

        # This branch treats Ge'ez as standard text (The Baseline)
        else:
            get_logger().info("Starting Baseline Training (No Splintering)...")
            text_processor = TextProcessorBaseline(language_utils)

        # 3. Write the processed text to D:\NLP 2026/... for SentencePiece to read
        save_corpus_as_text_file(text_processor, train_dataset_path, train_dataset_name)
        language_utils.save_additional_corpora_for_evaluation(text_processor)

    # 4. Train SentencePiece (BPE/Unigram) on the output of Step 3
    if get_run_params("TRAIN_TOKENIZERS"):
        tokenizer_corpus_path = get_corpus_path(get_corpus_name(train_dataset_path, train_dataset_name))
        for tokenizer_type in get_run_params("TOKENIZERS_TYPES"):
            for vocab_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
                tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
                train_tokenizer(tokenizer_type=tokenizer_type, vocab_size=vocab_size, input_path=tokenizer_corpus_path, output_path=tokenizer_path)

    # 5. Execute tokenization and static checks to compare Baseline vs. Splintered
    for tokenizer_type in get_run_params("TOKENIZERS_TYPES"):
        for vocab_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
            if get_run_params("TOKENIZE_CORPORA"):
                tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
                # Static check corpora (e.g., your combined Ge'ez file)
                for corpus_name in get_run_params("STATIC_CHECKS_CORPORA"):
                    corpus_path = get_corpus_path(corpus_name)
                    tokenized_corpus_path = get_tokenized_corpus_path(corpus_name, tokenizer_type, vocab_size)
                    
                    # Uses the tokenizer logic to process the final file
                    CorpusTokenizer().tokenize_corpus_into_file(tokenizer_path, corpus_path, tokenized_corpus_path)

            if get_run_params("RUN_STATIC_CHECKS"):
                get_logger().info(f"Running evaluation for {tokenizer_type} v{vocab_size}")
                types_length_distribution = run_types_length_distribution(tokenizer_type, vocab_size)
                add_static_result_to_file(types_length_distribution)

                for corpus_name in get_run_params("STATIC_CHECKS_CORPORA"):
                    corpus_path = get_corpus_path(corpus_name)
                    tokenized_corpus_path = get_tokenized_corpus_path(corpus_name, tokenizer_type, vocab_size)
                    tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
                    results = run_static_checks(corpus_path, tokenized_corpus_path, tokenizer_path, vocab_size)
                    add_static_result_to_file(results)

if __name__ == '__main__':
    # Support for Slurm (parallel runs) or single local execution
    slurm_array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')
    task_id = (int(slurm_array_task_id) - 1) if slurm_array_task_id else 0
    
    # Grab the current experiment (Baseline or All_Letters) from params.py
    experiment = experiments[task_id]
    experiment["TASK_ID"] = task_id
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