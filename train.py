# === train.py ===
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from model import CNNLSTM
from dataset import PerFrameLabeledDataset
from extract_frames import extract_frames_with_labels

# Detect MPS (Apple Silicon) or GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load video and frame-level labels
clips, labels = extract_frames_with_labels("data/match1.mp4", "data/match1.json")

# Create dataset
dataset = PerFrameLabeledDataset(clips, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Dataset sizes: Train = {len(train_dataset)}, Val = {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Initialize model and move to device
model = CNNLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("Model initialized and moved to device.")

# Training loop
for epoch in range(5):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = torch.stack(x_batch).to(device)  # [B, T, C, H, W]
        y_batch = torch.stack(y_batch).to(device)  # [B, T]
        outputs = model(x_batch)  # [B, T, num_classes]
        loss = criterion(outputs.view(-1, 2), y_batch.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = torch.stack(x_batch).to(device)
            y_batch = torch.stack(y_batch).to(device)
            outputs = model(x_batch)
            loss = criterion(outputs.view(-1, 2), y_batch.view(-1))
            val_loss += loss.item()
            predictions = torch.argmax(outputs, dim=2)  # [B, T]
            correct += (predictions == y_batch).sum().item()
            total += y_batch.numel()

    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
          f"Val Loss = {val_loss/len(val_loader):.4f}, "
          f"Val Acc = {correct/total:.2%}")
