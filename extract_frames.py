# === extract_frames.py ===
import cv2
import json
import torch
from torchvision import transforms
from utils import load_labels

# Extract frames and assign per-frame labels based on whether each timestamp is in a labeled segment
def extract_frames_with_labels(video_path, json_path, fps=5, max_frames=32):
    labels = load_labels(json_path)
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / orig_fps

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Store labeled time segments
    segments = labels['segments']

    # Create list of start-end pairs as tuples for in-play segments
    in_play_ranges = [(s['start'], s['end']) for s in segments]

    all_clips = []
    all_labels = []
    clip = []
    clip_labels = []

    for i in range(0, total_frames, int(orig_fps // fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = i / orig_fps

        # Check if current frame timestamp is in an in-play segment
        in_play = any(start <= timestamp <= end for start, end in in_play_ranges)
        label = 1 if in_play else 0

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = preprocess(frame)

        clip.append(frame)
        clip_labels.append(label)

        if len(clip) == max_frames:
            all_clips.append(torch.stack(clip))  # shape: [T, C, H, W]
            all_labels.append(torch.tensor(clip_labels))  # shape: [T]
            clip = []
            clip_labels = []

    cap.release()
    return all_clips, all_labels