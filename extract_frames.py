# === extract_frames.py ===

import cv2
import os
import json
import torch
from tqdm import tqdm 
from torchvision import transforms


def extract_and_save_frames_with_labels(
    video_path,
    json_path,
    output_dir="output",
    fps=5,
    normalize=True
):
    # Create subdirectories for saving
    frame_dir = os.path.join(output_dir, "frames")
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Load label segments from JSON
    with open(json_path, "r") as f:
        label_data = json.load(f)
    segments = label_data.get("segments", [])

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(round(orig_fps / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define image transform
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )
    preprocess = transforms.Compose(transform_list)

    # Iterate through frames
    idx = 0
    saved = 0
    pbar = tqdm(total=total_frames, desc="Extracting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        if idx % interval == 0:
            timestamp = idx / orig_fps  # seconds

            # Determine if timestamp is in an in-play segment
            in_play = any(seg["start"] <= timestamp <= seg["end"] for seg in segments)
            label = 1 if in_play else 0

            # Convert BGR (OpenCV) to RGB and apply transforms
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = preprocess(frame_rgb)  # [3, 224, 224]

            # Save frame and label
            torch.save(tensor, os.path.join(frame_dir, f"frame_{saved:05d}.pt"))
            torch.save(torch.tensor(label), os.path.join(label_dir, f"label_{saved:05d}.pt"))
            saved += 1

        idx += 1

    pbar.close()
    cap.release()
    print(f"✅ Done! Saved {saved} frames and labels to: {output_dir}/")


