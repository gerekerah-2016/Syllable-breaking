import sentencepiece as spm
import json
import os
import codecs

# --- PATHS ---
BASE_DIR = r"D:\NLP 2026\Splintering"
TRAIN_TEXT = os.path.join(BASE_DIR, "geez_training_data.csv") 
MAP_PATH = os.path.join(BASE_DIR, "experiments", "2025-01-21-geez-all_letters", "logs", "log-0-20251223-104553", "new_unicode_chars.json")

# 1. Load the splinters using a codec-safe approach
print("🔍 Attempting to load splinters...")

with codecs.open(MAP_PATH, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
    # The 'values' are the unique splinter characters
    protected_symbols = list(mapping.values())

# CRITICAL CHECK
sample = protected_symbols[0]
print(f"🛡️ Protecting {len(protected_symbols)} splinter symbols.")
print(f"DEBUG: First symbol raw: {repr(sample)}")

if "倀" in sample or "σ" in sample:
    print("❌ ERROR: Encoding is STILL broken. Do not proceed.")
    print("Check if the JSON file was created by a PowerShell script (which often uses UTF-16).")
else:
    print(f"✅ SUCCESS: Found valid symbol: {sample}")

    # 2. Train the Tokenizer
    spm.SentencePieceTrainer.train(
        input=TRAIN_TEXT,
        model_prefix='geez_splinter_tokenizer',
        vocab_size=3000,
        model_type='unigram',
        user_defined_symbols=protected_symbols, 
        character_coverage=1.0,
        input_sentence_size=100000,
        shuffle_input_sentence=True
    )
    print("🚀 New model 'geez_splinter_tokenizer.model' created successfully.")