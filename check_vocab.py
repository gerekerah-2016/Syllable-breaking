import sentencepiece as spm
import json
import os

# --- PATHS ---
SPM_MODEL = r"D:\NLP 2026\Splintering\experiments\2025-01-21-geez-all_letters\results\tokenizers\unigram_3000.model"
MAP_PATH = r"D:\NLP 2026\Splintering\experiments\2025-01-21-geez-all_letters\logs\log-0-20251223-104553\new_unicode_chars.json"

# Load Model
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)

# Load Mapping
with open(MAP_PATH, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
inverted_map = {v: k for k, v in mapping.items()}

def audit_vocabulary():
    vocab_size = sp.get_piece_size()
    print(f"--- Vocabulary Audit (Size: {vocab_size}) ---")
    
    splinter_count = 0
    standard_count = 0
    
    # We loop through every ID in the model's brain
    for i in range(vocab_size):
        piece = sp.id_to_piece(i).replace(' ', '')
        # Check if this piece maps back to a splinter (e.g., 0:ተ)
        original_form = inverted_map.get(piece, piece)
        
        if ":" in original_form:
            splinter_count += 1
            if splinter_count <= 10: # Print the first 10 we find
                print(f"Found Splinter: ID {i} -> {original_form}")
        else:
            standard_count += 1

    print("\n--- Summary ---")
    print(f"Total Splinters (index:char): {splinter_count}")
    print(f"Total Standard Tokens:        {standard_count}")
    
    if splinter_count == 0:
        print("\n❌ WARNING: No splinters found! Your tokenizer training ignored them.")
    elif splinter_count < 100:
        print("\n⚠️ ALERT: Low splinter count. The AI might prefer plain letters instead.")
    else:
        print("\n✅ SUCCESS: Vocabulary contains enough splinters for morphological learning.")

audit_vocabulary()