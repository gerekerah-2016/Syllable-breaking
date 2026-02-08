import sentencepiece as spm
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SETTINGS ---
BASE_DIR = r"D:\NLP 2026\Splintering"
MODEL_PATH = os.path.join(BASE_DIR, "experiments", "2025-01-21-geez-all_letters", "results", "tokenizers", "unigram_3000.model")

# Define search paths for the dataset
possible_inputs = [
    os.path.join(BASE_DIR, "data", "word_dict", "train_combined.json"),
    os.path.join(BASE_DIR, "Geez-Dataset", "local_Geez_Dataset.json"),
    os.path.join(BASE_DIR, "Geez-Dataset", "Geez_Dataset_default.json")
]

INPUT_FILE = None
for path in possible_inputs:
    if os.path.exists(path):
        INPUT_FILE = path
        break

if not INPUT_FILE:
    print("❌ Could not find the dataset file! Please check your paths.")
    exit()

print(f"🚀 Using dataset: {INPUT_FILE}")

# --- 2. DATA LOADING ---
lines = []
if INPUT_FILE.endswith('.json'):
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # Extract only the actual text sentences, ignore tiny mapping keys
        # We assume the longest strings are our sentences
        sample_val = next(iter(data.values()))
        if isinstance(sample_val, str) and len(sample_val) > 2:
            lines = list(data.values())
        else:
            lines = list(data.keys())
    elif isinstance(data, list):
        lines = [str(x) for x in data]
else:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

# Remove any empty or very short noise
lines = [line for line in lines if len(line) > 5]
print(f"📄 Loaded {len(lines)} lines of text.")

# --- 3. TOKENIZATION ---
sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
all_ids = sp.encode_as_ids(lines)

df = pd.DataFrame({
    "length": [len(ids) for ids in all_ids]
})

# --- 4. VISUALIZATION ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.histplot(df['length'], kde=True, color='#2c7fb8', bins=50)

mean_len = df['length'].mean()
p95_len = df['length'].quantile(0.95)

plt.axvline(mean_len, color='red', linestyle='--', label=f'Avg: {mean_len:.1f}')
plt.axvline(p95_len, color='orange', linestyle='--', label=f'95% Cutoff: {p95_len:.1f}')
plt.title('Ge\'ez Token Sequence Length Distribution (Splintered)')
plt.xlabel('Number of Tokens')
plt.ylabel('Count')
plt.legend()
plt.show()

print(f"✅ Success! 95% of sequences are under {p95_len:.1f} tokens.")