from extract_clips import extract_clips_with_labels
from extract_frames import extract_and_save_frames_with_labels
import torch

#extract_clips_with_labels("data/match1.mp4", "data/match1.json", "data/match1_clips", fps=4, window_size=16, stride=6)

extract_and_save_frames_with_labels(
    video_path="data/match1.mp4",
    json_path="data/match1.json",
    output_dir= '/content/local_dataset/match1_frames', #"data/match1_frames",
    fps= 5,
    normalize= True
)
