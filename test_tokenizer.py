import sentencepiece as spm
import json
import os
import glob

BASE_DIR = r"D:\NLP 2026\Splintering"
MAP_PATH = r"D:\NLP 2026\Splintering\experiments\2025-01-21-geez-all_letters\logs\log-0-20251223-104553\new_unicode_chars.json"

# 1. LOAD MAP
with open(MAP_PATH, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
inverted_map = {v: k for k, v in mapping.items()}

# 2. LOCATE THE TOKENIZERS FOLDER
# We'll look for the folder containing both unigram and bpe models
search_pattern = os.path.join(BASE_DIR, "experiments", "**", "results", "tokenizers")
tokenizer_dirs = glob.glob(search_pattern, recursive=True)

if not tokenizer_dirs:
    print("❌ Could not find the results/tokenizers folder!")
    exit()

# Use the most recent results folder
tokenizer_dirs.sort(key=os.path.getmtime, reverse=True)
TOKENIZER_PATH = tokenizer_dirs[0]

unigram_model = os.path.join(TOKENIZER_PATH, "unigram_3000.model")
bpe_model = os.path.join(TOKENIZER_PATH, "bpe_3000.model")


def get_splits_with_ids(model_path, text):
    if not os.path.exists(model_path):
        return "Model not found", []
    
    sp = spm.SentencePieceProcessor(model_file=model_path)
    # Get both the pieces (text) and the IDs (numbers)
    pieces = sp.encode_as_pieces(text)
    ids = sp.encode_as_ids(text)
    
    decoded_pieces = []
    for piece in pieces:
        # Convert PUA back to readable Ge'ez
        readable = "".join([inverted_map.get(c, c).split(':')[-1] for c in piece])
        decoded_pieces.append(readable)
    
    return decoded_pieces, ids

# --- RUN THE COMPARISON ---
test_sentence = "በመጀመሪያ ቃል ነበረ፤ ያም ቃል በእግዚአብሔር ዘንድ ነበረ፤"

bpe_pieces, bpe_ids = get_splits_with_ids(bpe_model, test_sentence)
uni_pieces, uni_ids = get_splits_with_ids(unigram_model, test_sentence)

print(f"BPE IDs:     {bpe_ids}")
print(f"Unigram IDs: {uni_ids}")
print("-" * 30)
print(f"BPE Splits:     {' | '.join(bpe_pieces)}")
print(f"Unigram Splits: {' | '.join(uni_pieces)}")

# Check for equality
if bpe_ids == uni_ids:
    print("\n⚠️ WARNING: Both models produced IDENTICAL token IDs.")
else:
    print("\n✅ SUCCESS: The models are mathematically different, even if the splits look similar.")