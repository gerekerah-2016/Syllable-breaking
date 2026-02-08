import pandas as pd
import sentencepiece as spm
import os
import json

# --- SETTINGS ---
BASE_DIR = r"D:\NLP 2026\Splintering"
SPM_MODEL = os.path.join(BASE_DIR, "experiments", "2025-01-21-geez-all_letters", "results", "tokenizers", "unigram_3000.model")
OUTPUT_CSV = "geez_training_data.csv"

# Potential data locations we've seen in your logs
possible_inputs = [
    os.path.join(BASE_DIR, "data", "word_dict", "train_combined.json"),
    os.path.join(BASE_DIR, "Geez-Dataset", "local_Geez_Dataset.json")
]

# 1. Find the file
INPUT_FILE = None
for path in possible_inputs:
    if os.path.exists(path):
        INPUT_FILE = path
        break

if not INPUT_FILE:
    print("❌ Could not find your dataset JSON! Please check the folder.")
    exit()

# 2. Load Tokenizer
if not os.path.exists(SPM_MODEL):
    print(f"❌ Tokenizer model not found at: {SPM_MODEL}")
    exit()
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)
# ... (Keep the imports and path settings the same) ...

# 3. Extract Text from JSON
print(f"📂 Loading data from: {INPUT_FILE}")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = []
if isinstance(data, dict):
    # We'll take both keys and values, but filter for actual Ge'ez content later
    # This ensures we don't miss the data if it's stored 'backwards'
    for k, v in data.items():
        lines.append(str(k))
        lines.append(str(v))
else:
    lines = [str(x) for x in data]

# 4. Tokenize and Save
print(f"✂️ Processing {len(lines)} potential text strings...")
data_rows = []
for line in lines:
    line = line.strip()
    # Lowered the threshold: If it's at least 2 characters, try to tokenize it
    if len(line) >= 2: 
        ids = sp.encode_as_ids(line)
        if ids: # Only add if the tokenizer actually produced IDs
            data_rows.append({'token_ids': str(ids)})

if not data_rows:
    print("❌ Still 0 rows! Let's print a sample of the JSON to see what's wrong:")
    print(list(data.items())[:2]) # Print first two items for debugging
    exit()

df = pd.DataFrame(data_rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Success! Created {OUTPUT_CSV} with {len(df)} rows.")
