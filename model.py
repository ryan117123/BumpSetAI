# === model.py ===
import torch
import torch.nn as nn
from torchvision import models

# CNN + LSTM architecture with frozen CNN encoder
def get_resnet18_backbone():
    resnet = models.resnet18(pretrained=True)
    return nn.Sequential(*list(resnet.children())[:-1])

class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, num_classes=2):
        super().__init__()
        self.cnn = get_resnet18_backbone()
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        out = self.classifier(lstm_out[:, -1, :])
        return out
