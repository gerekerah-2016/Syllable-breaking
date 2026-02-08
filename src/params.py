import copy
from datetime import datetime, timezone

# =====================
# Runtime parameters
# =====================

run_params = {}  # filled by set_run_params in main

def set_run_params(params):
    global run_params
    run_params = params
    run_params["TIMESTAMP"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

def get_run_params(key):
    return run_params[key]

def get_all_run_params():
    return run_params

# =====================
# Language configuration
# =====================

LANGUAGE = 'gez'          # Geez ISO code
LANGUAGE_FULL = 'geez'
EXPERIMENT_DATE = '2025-01-21'

# Geez → no Hebrew static checks
STATIC_CHECKS_CORPORA = []

# =====================
# Base experiment template
# =====================
# =====================
# Base experiment template
# =====================
# In params.py
experiment_template = {
    'LANGUAGE': 'gez',
    'SAVE_CORPORA_INTO_FILE': True,
    
    # ❌ DON'T USE: 'Geez-Dataset'
    # ✅ USE THE FULL PATH:
    'SPLINTER_TRAINING_CORPUS_PATH': r'D:\NLP 2026\Splintering\Geez-Dataset', 
 
    'SPLINTER_TRAINING_CORPUS_NAME': 'default',
    'TRAIN_TOKENIZERS': True,
    'TOKENIZERS_TYPES': ['unigram', 'bpe'],
    
    "TOKENIZE_CORPORA": True,
    "RUN_STATIC_CHECKS": True,
    "STATIC_CHECKS_CORPORA": ["default"], 
    "EXPERIMENT_NAME": "2025-01-21-geez-all_letters",
}
{
    # ... existing params ...
    "TOKENIZE_CORPORA": True,
    "RUN_STATIC_CHECKS": True,
    "STATIC_CHECKS_CORPORA": ["default"], 
    "EXPERIMENT_NAME": "2025-01-21-geez-all_letters",
}

# =====================
# Experiment variants
# =====================

def get_all_letters_template():
    t = copy.deepcopy(experiment_template)
    t['EXPERIMENT_NAME'] = f'{EXPERIMENT_DATE}-{LANGUAGE_FULL}-all_letters'
    # --- SPLINTERED VERSION ---
    # This triggers GeezUtils.replace_final_letters (Breaking syllables)
    t['IS_ENCODED'] = True 
    t['SPLINTER_LETTERS_SUBSET'] = None
    return t

def get_baseline_template():
    t = copy.deepcopy(experiment_template)
    t['EXPERIMENT_NAME'] = f'{EXPERIMENT_DATE}-{LANGUAGE_FULL}-baseline'
    # --- RAW VERSION ---
    # This uses the original Ge'ez syllables for comparison
    t['IS_ENCODED'] = False
    t['SPLINTER_LETTERS_SUBSET'] = None
    return t

# =====================
# Split into multiple runs
# =====================

def split_to_separate_runs(template):
    runs = []
    # Testing smaller vocab sizes to see if Splintering captures roots better
    vocab_sizes = [3000, 8000, 4000, 2000, 1000, 800]
    for vocab_size in vocab_sizes:
        run = copy.deepcopy(template)
        run['TOKENIZERS_VOCAB_SIZES'] = [vocab_size]
        runs.append(run)
    return runs

def get_dummy_experiment(experiment_name: str):
    experiment = copy.deepcopy(get_baseline_template())
    experiment['EXPERIMENT_NAME'] = experiment_name
    experiment["TASK_ID"] = '1000000'
    experiment['TOKENIZERS_VOCAB_SIZES'] = [128000]
    return experiment

# =====================
# Final experiments list
# =====================

experiments = []
# 1. Runs the Splintered Ge'ez pipeline
experiments.extend(split_to_separate_runs(get_all_letters_template()))
# 2. Runs the standard Ge'ez pipeline
experiments.extend(split_to_separate_runs(get_baseline_template()))