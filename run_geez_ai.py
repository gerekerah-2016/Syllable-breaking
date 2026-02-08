import torch
import sentencepiece as spm
from train_geez import GeezPredictor # Importing your class

# Settings
MODEL_PATH = "geez_model.pth"
SPM_MODEL = r"D:\NLP 2026\Splintering\experiments\2025-01-21-geez-all_letters\results\tokenizers\unigram_3000.model"

def reconstruct_geez(splinter_list):
    """Reassembles splinters into words based on the SPLINTER paper logic."""
    # This logic reverses the iterative pruning described in the paper
    base_chars = []
    reductions = []
    
    for token in splinter_list:
        if ":" in token:
            try:
                idx, char = token.split(":")
                reductions.append((int(idx), char))
            except: continue
        else:
            base_chars.append(token)
            
    word = list("".join(base_chars))
    # Apply reductions in reverse order of their removal (ascending index)
    reductions.sort(key=lambda x: x[0])
    for idx, char in reductions:
        word.insert(max(0, min(idx, len(word))), char)
    return "".join(word)

# Load Model
device = torch.device("cpu")
model = GeezPredictor(vocab_size=3000)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load Tokenizer
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)

def generate(start_text, max_len=5):
    ids = sp.encode_as_ids(start_text)
    input_tensor = torch.tensor([ids])
    
    generated_ids = ids
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(input_tensor)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated_ids.append(next_token)
            input_tensor = torch.tensor([generated_ids])
            
    tokens = [sp.id_to_piece(i) for i in generated_ids]
    return reconstruct_geez(tokens)

print(f"AI Prediction: {generate('በ')}")