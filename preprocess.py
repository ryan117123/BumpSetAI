from extract_frames import extract_frames_with_labels
import torch

extract_frames_with_labels("data/match1.mp4", "data/match1.json", "data/match1_clips", fps=4, window_size=16, stride=6)
