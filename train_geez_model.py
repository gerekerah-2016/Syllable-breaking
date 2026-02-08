import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# --- 1. Your Model ---
class GeezPredictor(nn.Module):
    def __init__(self, vocab_size=3000, embedding_dim=128, hidden_dim=256):
        super(GeezPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        return self.fc(output)

# --- 2. Data Loader ---
class GeezDataset(Dataset):
    def __init__(self, csv_file, max_len=16):
        self.df = pd.read_csv(csv_file)
        self.max_len = max_len
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        token_ids = eval(self.df.iloc[idx]['token_ids'])
        token_ids = token_ids[:self.max_len]
        padding = [0] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids + padding, dtype=torch.long)

# --- 3. Run Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = GeezDataset("geez_training_data.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = GeezPredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

print(f"Starting training on {device}...")

for epoch in range(5): # Let's start with 5 epochs
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        # We predict the 'next' splinter, so shift the data
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, 3000), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

# Save the trained brain
torch.save(model.state_dict(), "geez_model.pth")