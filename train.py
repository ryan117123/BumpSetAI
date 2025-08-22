# === train.py ===
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torchvision import transforms
from model import CNNLSTM

# === FrameClipDataset that loads all frames once and builds clips dynamically ===
class FrameClipDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, label_dir, clip_len=16, stride=4):
        self.clip_len = clip_len
        self.stride = stride

        # Load all frames and labels into memory
        self.frames = []
        self.labels = []
        for fname in sorted(os.listdir(frame_dir)):
            self.frames.append(torch.load(os.path.join(frame_dir, fname)))  # [3, 224, 224]
        for fname in sorted(os.listdir(label_dir)):
            self.labels.append(torch.load(os.path.join(label_dir, fname)).item())  # scalar

        self.frames = torch.stack(self.frames)  # [N, 3, 224, 224]
        self.labels = torch.tensor(self.labels)  # [N]

        # Build list of valid clip start indices
        self.clip_starts = [i for i in range(0, len(self.frames) - clip_len + 1, stride)]

    def __len__(self):
        return len(self.clip_starts)

    def __getitem__(self, idx):
        start = self.clip_starts[idx]
        end = start + self.clip_len
        clip = self.frames[start:end]  # [T, C, H, W]
        label = self.labels[start:end]  # [T]
        return clip, label

# Detect device (GPU on Colab or fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load full frame dataset into RAM and build clips dynamically
frame_dir = "/content/local_dataset/frames"
label_dir = "/content/local_dataset/labels"
dataset = FrameClipDataset(frame_dir, label_dir, clip_len=16, stride=4)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Initialize model and move to device
model = CNNLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

import time
# Training loop
for epoch in range(5):
    model.train()
    train_loss = 0
    start = time.time()
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)  # [B, T, C, H, W]
        y_batch = y_batch.to(device)  # [B, T]
        outputs = model(x_batch)      # [B, T, num_classes]
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_batch.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i == 0:
            end = time.time()
            print(f"Time for first batch: {end - start:.2f} seconds")
            start = end

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_batch.reshape(-1))
            val_loss += loss.item()
            predictions = torch.argmax(outputs, dim=2)  # [B, T]
            correct += (predictions == y_batch).sum().item()
            total += y_batch.numel()
    print("epoch", epoch+1)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
          f"Val Loss = {val_loss/len(val_loader):.4f}, "
          f"Val Acc = {correct/total:.2%}")
