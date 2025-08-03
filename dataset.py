# === dataset.py ===
from torch.utils.data import Dataset

# Dataset returning a clip with per-frame labels
class PerFrameLabeledDataset(Dataset):
    def __init__(self, clips, clip_labels):
        self.clips = clips
        self.labels = clip_labels

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return self.clips[idx], self.labels[idx]  # ([T, C, H, W], [T])
