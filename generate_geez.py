import torch
import sentencepiece as spm
import json
import os

# --- 1. SETUP ---
BASE_DIR = r"D:\NLP 2026\Splintering"
MAP_PATH = r"D:\NLP 2026\Splintering\experiments\2025-01-21-geez-all_letters\logs\log-0-20251223-104553\new_unicode_chars.json"
MODEL_BIN = "geez_model.pth" # The weights you just trained
SPM_MODEL = r"D:\NLP 2026\Splintering\experiments\2025-01-21-geez-all_letters\results\tokenizers\unigram_3000.model"

# Load Mappings
with open(MAP_PATH, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
inverted_map = {v: k for k, v in mapping.items()}

# Load SentencePiece to handle IDs
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)

# --- 2. THE BRAIN (Architecture must match training) ---
class GeezPredictor(torch.nn.Module):
    def __init__(self, vocab_size=3000, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.gru = torch.nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        return self.fc(output)

# Load Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeezPredictor().to(device)
model.load_state_dict(torch.load(MODEL_BIN, map_location=device))
model.eval()

# --- 3. GENERATION FUNCTION ---
def generate_text(start_text, max_gen=10):
    print(f"Seed: {start_text}", end=" ", flush=True)
    
    # 1. Encode the seed text to IDs
    current_ids = sp.encode_as_ids(start_text)
    input_tensor = torch.tensor([current_ids]).to(device)
    
    generated_output = []
    
    for _ in range(max_gen):
        with torch.no_grad():
            logits = model(input_tensor)
            # Take the prediction for the very last token
            next_id = torch.argmax(logits[0, -1]).item()
            
            # 2. Map ID -> Piece -> Readable Ge'ez
            piece = sp.id_to_piece(next_id)
            readable = "".join([inverted_map.get(c, c).split(':')[-1] for c in piece])
            
            print(f"| {readable}", end=" ", flush=True)
            
            # 3. Update input for next step
            next_id_tensor = torch.tensor([[next_id]]).to(device)
            input_tensor = torch.cat([input_tensor, next_id_tensor], dim=1)
            
    print("\nDone!")

# --- 4. TEST IT ---
generate_text("በመጀመሪያ")
def reconstruct_word(splintered_tokens):
    """
    Takes a list of predicted tokens and reassembles them.
    Example input: ['ደወለ', '0:ተ', '3:ያ'] -> Result: 'ተደወያለ'
    """
    # 1. Separate the base root from the composite splinters
    base_chars = ""
    reductions = []
    
    for token in splintered_tokens:
        if ":" in token:
            # It's a splinter (e.g., "0:ተ")
            idx_part, char = token.split(":")
            # Handle potential non-numeric indices (paper mentions negative indices)
            idx = int(idx_part)
            reductions.append((idx, char))
        else:
            # It's the root or a standard subword
            base_chars += token

    # 2. Re-insert splinters into the base at their specific indices
    word_list = list(base_chars)
    
    # Sort by index to ensure correct insertion order
    reductions.sort(key=lambda x: x[0])
    
    for idx, char in reductions:
        # Safety check: ensure index is within current word bounds
        actual_idx = max(0, min(idx, len(word_list)))
        word_list.insert(actual_idx, char)
            
    return "".join(word_list)