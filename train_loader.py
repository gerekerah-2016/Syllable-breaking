import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class GeezDataset(Dataset):
    def __init__(self, csv_file, max_len=16):
        # Load the CSV we created in the last step
        self.df = pd.read_csv(csv_file)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Convert the string representation of list back to actual list of ints
        token_ids = eval(self.df.iloc[idx]['token_ids'])
        
        # 1. TRUNCATE: If it's too long, cut it
        token_ids = token_ids[:self.max_len]
        
        # 2. PADDING: If it's too short, add 0s until it's 16 tokens long
        padding_len = self.max_len - len(token_ids)
        padded_ids = token_ids + ([0] * padding_len)
        
        return torch.tensor(padded_ids, dtype=torch.long)

# --- INITIALIZE THE LOADER ---
CSV_PATH = "geez_training_data.csv"
geez_data = GeezDataset(CSV_PATH, max_len=16)

# Batch size 32 means we feed 32 sentences at a time to the GPU
train_loader = DataLoader(geez_data, batch_size=32, shuffle=True)

# Test run: grab one batch
first_batch = next(iter(train_loader))
print(f"Batch Shape: {first_batch.shape}") # Should be [32, 16]
print(f"First sequence in batch:\n{first_batch[0]}")