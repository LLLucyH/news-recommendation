import torch.optim as optim
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        # Move data to GPU/CPU
        hist = batch['history'].to(device)
        cand = batch['candidate'].to(device)
        label = batch['label'].to(device)
        
        # Forward pass
        preds = model(hist, cand)
        
        # Calculate loss (Binary Cross Entropy)
        loss = criterion(preds, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

# Example Initialization logic
def main():
    from config import Config
    from data_processor import MindProcessor
    from dataset import MINDDataset
    from torch.utils.data import DataLoader

    cfg = Config()
    
    # 1. Process Data
    processor = MindProcessor(cfg)
    processor.build_dictionaries(cfg.TRAIN_NEWS)
    
    # 2. Create Dataset
    train_ds = MINDDataset(cfg.TRAIN_BEHAVIORS, processor, cfg, mode='train')
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    # 3. Initialize Model
    model = TwoTowerRanker(cfg, len(processor.word_dict)).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # 4. Loop
    for epoch in range(cfg.EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.DEVICE)
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} - Loss: {avg_loss:.4f}")