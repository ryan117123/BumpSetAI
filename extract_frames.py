# === extract_frames.py ===
import cv2
import json
import torch
import os
import numpy as np
from torchvision import transforms
from utils import load_labels

# Extract frames using a sliding window and save each clip+label to disk as uint8

def extract_frames_with_labels(video_path, json_path, output_dir, fps=5, window_size=20, stride=5):
    labels = load_labels(json_path)
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / orig_fps

    os.makedirs(output_dir, exist_ok=True)
    clips_dir = os.path.join(output_dir, "clips")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    segments = labels['segments']
    in_play_ranges = [(s['start'], s['end']) for s in segments]

    frames = []
    frame_labels = []

    for i in range(0, total_frames, int(orig_fps // fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = i / orig_fps

        in_play = any(start <= timestamp <= end for start, end in in_play_ranges)
        label = 1 if in_play else 0

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))  # Resize before saving

        frames.append(frame.astype(np.uint8))  # Save as uint8
        frame_labels.append(label)

    cap.release()

    clip_count = 0
    for i in range(0, len(frames) - window_size + 1, stride):
        clip = np.stack(frames[i:i + window_size])        # shape: [T, H, W, C]
        label_seq = torch.tensor(frame_labels[i:i + window_size])  # shape: [T]

        torch.save(torch.from_numpy(clip), os.path.join(clips_dir, f"clip_{clip_count}.pt"))
        torch.save(label_seq, os.path.join(labels_dir, f"label_{clip_count}.pt"))
        clip_count += 1

    print(f"Saved {clip_count} uint8 clips to {output_dir}")
    return clip_count
