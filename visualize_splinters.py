import torch
import torch.nn.functional as F
import sentencepiece as spm
import json

# --- 1. SETUP ---
MODEL_BIN = "geez_model.pth"
SPM_MODEL = r"D:\NLP 2026\Splintering\experiments\2025-01-21-geez-all_letters\results\tokenizers\unigram_3000.model"
MAP_PATH = r"D:\NLP 2026\Splintering\experiments\2025-01-21-geez-all_letters\logs\log-0-20251223-104553\new_unicode_chars.json"

with open(MAP_PATH, 'r', encoding='utf-8') as f:
    inverted_map = {v: k for k, v in json.load(f).items()}
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)

# --- 2. LOAD BRAIN ---
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

model = GeezPredictor()
model.load_state_dict(torch.load(MODEL_BIN, map_location="cpu"))
model.eval()

# --- 3. THE VISUALIZER ---
def see_inside_ai(root_text):
    print(f"\n🔍 Analyzing Template Map for Root: {root_text}")
    ids = sp.encode_as_ids(root_text)
    input_tensor = torch.tensor([ids])
    
    with torch.no_grad():
        logits = model(input_tensor)[0, -1]
        probs = F.softmax(logits, dim=-1)
        
        # Get Top 5 Predictions
        top_probs, top_ids = torch.topk(probs, 10)
        
    print(f"{'Splinter':<15} | {'Confidence':<10} | {'Meaning'}")
    print("-" * 45)
    
    for i in range(len(top_ids)):
        p_id = top_ids[i].item()
        prob = top_probs[i].item()
        piece = sp.id_to_piece(p_id).replace(' ', '')
        readable = inverted_map.get(piece, piece)
        
        # Explain what the AI is doing
        action = "Next Part/Word"
        if ":" in readable:
            action = f"Insert '{readable.split(':')[1]}' at pos {readable.split(':')[0]}"
            
        print(f"{readable:<15} | {prob*100:>8.2f}%    | {action}")

# TEST IT
see_inside_ai("መረፀ")
see_inside_ai("ቀደሰ")