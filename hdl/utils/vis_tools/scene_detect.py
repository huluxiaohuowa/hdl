import os
import json
import base64
from pathlib import Path
from PIL import Image
import openai
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
from tqdm import tqdm
import pandas as pd
import subprocess


def detect_scenes_cli(video_path, output_dir, threshold=20):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    # output_dir = video_path.parent
    base_name = video_path.stem
    csv_path = output_dir / f"{base_name}-Scenes.csv"

    subprocess.run([
        "scenedetect",
        "-i", str(video_path),
        "detect-content",
        f"--threshold={threshold}",
        "list-scenes",
        "-o", str(output_dir)
    ], check=True)

    return csv_path

def read_start_frames_from_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=1)
    return df['Start Frame'].astype(int).tolist()


class SceneDetector(object):
