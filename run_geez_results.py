import torch
import sentencepiece as spm
import json
import os

# --- 1. SETUP ---
BASE_DIR = r"D:\NLP 2026\Splintering"
MAP_PATH = os.path.join(BASE_DIR, "experiments", "2025-01-21-geez-all_letters", "logs", "log-0-20251223-104553", "new_unicode_chars.json")
MODEL_BIN = "geez_model.pth"
SPM_MODEL = os.path.join(BASE_DIR, "experiments", "2025-01-21-geez-all_letters", "results", "tokenizers", "unigram_3000.model")

with open(MAP_PATH, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
inverted_map = {v: k for k, v in mapping.items()}
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)

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

device = torch.device("cpu")
model = GeezPredictor()
model.load_state_dict(torch.load(MODEL_BIN, map_location=device))
model.eval()

# --- 2. RECONSTRUCTION ---
def reconstruct_word(splintered_tokens):
    base_chars = ""
    reductions = []
    for token in splintered_tokens:
        clean_token = token.replace(' ', '')
        actual_content = inverted_map.get(clean_token, clean_token)
        if ":" in actual_content:
            try:
                idx_part, char = actual_content.split(":")
                reductions.append((int(idx_part), char))
            except: continue
        else:
            base_chars += actual_content

    word_list = list(base_chars)
    reductions.sort(key=lambda x: x[0])
    for idx, char in reductions:
        actual_idx = max(0, min(idx, len(word_list)))
        word_list.insert(actual_idx, char)
    return "".join(word_list).replace('▁', '')

# --- 3. THE GENERATE FUNCTION ---
def generate(seed_text, max_gen=5):
    ids = sp.encode_as_ids(seed_text)
    input_tensor = torch.tensor([ids])
    
    generated_ids = []
    for _ in range(max_gen):
        with torch.no_grad():
            logits = model(input_tensor)
            # We pick the best next token
            next_id = torch.argmax(logits[0, -1]).item()
            
            # Ge'ez Filter: Stop if the AI tries to use non-Ge'ez junk
            piece = sp.id_to_piece(next_id)
            is_geez = any('\u1200' <= c <= '\u137F' for c in piece)
            if not is_geez and ":" not in piece:
                break
                
            generated_ids.append(next_id)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_id]])], dim=1)

    all_pieces = [sp.id_to_piece(i) for i in ids + generated_ids]
    result = reconstruct_word(all_pieces)
    print(f"\n[Prompt]: {seed_text}")
    print(f"[Result]: {result}")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    generate("ኣርድእት")
    generate("በመጀመር")