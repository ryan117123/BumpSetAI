# === utils.py ===
import json

# Load annotation JSON file with segment start/end timestamps
def load_labels(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# === model.py ===
import torch
import torch.nn as nn
from torchvision import models

# CNN + LSTM architecture with frozen CNN encoder
def get_resnet18_backbone():
    resnet = models.resnet18(pretrained=True)
    # Remove final FC layer to get 512-dim feature vector
    return nn.Sequential(*list(resnet.children())[:-1])

class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, num_classes=2):
        super().__init__()
        self.cnn = get_resnet18_backbone()
        for param in self.cnn.parameters():
            param.requires_grad = False  # Freeze CNN

        # LSTM to process sequence of frame features
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)  # Final prediction layer

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Flatten batch and time
        feats = self.cnn(x).squeeze(-1).squeeze(-1)  # [B*T, 512]
        feats = feats.view(B, T, -1)  # [B, T, 512]
        lstm_out, _ = self.lstm(feats)
        out = self.classifier(lstm_out[:, -1, :])  # Use last LSTM output
        return out
