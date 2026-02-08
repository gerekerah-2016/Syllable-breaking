import sentencepiece as spm
import pandas as pd
import os

# 1. SETUP
MODEL_PATH = r"D:\NLP 2026\Splintering\experiments\2025-01-21-geez-all_letters\results\tokenizers\unigram_3000.model"
INPUT_FILE = r"D:\NLP 2026\Splintering\Geez-Dataset\default.txt" # Change to your source file
OUTPUT_FILE = "geez_training_data.csv"

sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)

def process_batch(input_path):
    print(f"🚀 Processing {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # SentencePiece can process the whole list at once (very fast)
    all_ids = sp.encode_as_ids(lines)
    
    # Create a structured dataset
    data = {
        "text": lines,
        "token_ids": all_ids,
        "length": [len(ids) for ids in all_ids]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"✅ Saved {len(df)} sequences to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_batch(INPUT_FILE)