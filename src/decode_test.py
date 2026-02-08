import sentencepiece as spm
import json
import os

# 1. Load the inversion map generated during training
# This is usually saved in your data or experiment directory
MAP_PATH = "./data/word_dict/new_unicode_chars_inverted_map.json" 

with open(MAP_PATH, 'r', encoding='utf-8') as f:
    inverted_map = json.load(f)

# 2. Load your trained model
model_path = "./models/unigram_3000.model"
sp = spm.SentencePieceProcessor(model_file=model_path)

def decode_splinters(text):
    # Encode text into tokens
    tokens = sp.encode_as_pieces(text)
    
    decoded_output = []
    for token in tokens:
        # Check if the token (or part of it) is a PUA character in our map
        # Note: SentencePiece might combine multiple PUA chars into one token
        decoded_token = ""
        for char in token:
            if char in inverted_map:
                # The map stores '0:መ', we just want the 'መ' part
                splinter_val = inverted_map[char].split(':')[-1]
                decoded_token += splinter_val
            else:
                decoded_token += char
        decoded_output.append(decoded_token)
    
    return decoded_output

# 3. Test it!
test_word = "በመጀመሪያ"
result = decode_splinters(test_word)

print(f"Original: {test_word}")
print(f"Splintered Tokens: {' | '.join(result)}")