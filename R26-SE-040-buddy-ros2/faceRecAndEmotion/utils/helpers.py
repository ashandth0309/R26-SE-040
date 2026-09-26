"""
Helper functions for faceRecAndEmotion modules.
"""

import json
import os
from datetime import datetime

import cv2


def load_json(file_path, default=None):
    """
    Load JSON data from file.
    Returns default if file does not exist or is invalid.
    """
    if default is None:
        default = {}

    try:
        if not os.path.exists(file_path):
            return default

        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as error:
        print(f"⚠️ Could not load JSON {file_path}: {error}")
        return default


def save_json(file_path, data):
    """
    Save JSON data to file.
    Creates parent folders automatically.
    """
    try:
        folder = os.path.dirname(str(file_path))

        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        return True

    except Exception as error:
        print(f"⚠️ Could not save JSON {file_path}: {error}")
        return False


def get_timestamp():
    """
    Return current timestamp string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_file_timestamp():
    """
    Return timestamp safe for filenames.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_dir(path):
    """
    Create directory if missing.
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_frame(frame, directory, prefix="frame"):
    """
    Save an OpenCV frame image to a directory.
    Returns saved file path, or None if failed.
    """
    try:
        ensure_dir(directory)

        filename = f"{prefix}_{get_file_timestamp()}.jpg"
        file_path = os.path.join(str(directory), filename)

        success = cv2.imwrite(file_path, frame)

        if success:
            return file_path

        return None

    except Exception as error:
        print(f"⚠️ Could not save frame: {error}")
        return None