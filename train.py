# === train.py ===
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torchvision import transforms
from model import CNNLSTM

# Custom Dataset class to load saved uint8 .pt files and apply preprocessing
class DiskClipDataset(torch.utils.data.Dataset):
    def __init__(self, clips_dir, labels_dir):
        self.clips_dir = clips_dir
        self.labels_dir = labels_dir
        self.filenames = sorted(os.listdir(clips_dir))

        # Define transform (to match frozen CNN input)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),  # Converts HWC uint8 [0-255] to CHW float32 [0-1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        clip_path = os.path.join(self.clips_dir, self.filenames[idx])
        label_path = os.path.join(self.labels_dir, self.filenames[idx].replace("clip", "label"))

        clip = torch.load(clip_path).numpy()  # shape: [T, H, W, C], dtype=uint8
        label = torch.load(label_path)        # shape: [T]

        # Apply transforms to each frame
        processed_frames = [self.preprocess(clip[i]) for i in range(clip.shape[0])]
        clip_tensor = torch.stack(processed_frames)  # shape: [T, C, H, W]

        return clip_tensor, label

# Detect device (GPU on Colab or fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset from saved .pt files
clips_dir = "/content/drive/MyDrive/BumpSetAI_data/match1_clips/clips"
labels_dir = "/content/drive/MyDrive/BumpSetAI_data/match1_clips/labels"
dataset = DiskClipDataset(clips_dir, labels_dir)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Initialize model and move to device
model = CNNLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(5):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)  # [B, T, C, H, W]
        y_batch = y_batch.to(device)  # [B, T]
        outputs = model(x_batch)      # [B, T, num_classes]
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_batch.reshape(-1))
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
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_batch.reshape(-1))
            val_loss += loss.item()
            predictions = torch.argmax(outputs, dim=2)  # [B, T]
            correct += (predictions == y_batch).sum().item()
            total += y_batch.numel()

    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
          f"Val Loss = {val_loss/len(val_loader):.4f}, "
          f"Val Acc = {correct/total:.2%}")
