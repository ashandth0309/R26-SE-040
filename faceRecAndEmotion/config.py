"""
Configuration file for Smart AI Dog Robot
Optimized version (safe changes only)
"""

import os
import cv2

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATABASE_DIR = BASE_DIR / "database"
UNKNOWN_FRAMES_DIR = BASE_DIR / "unknown_frames"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FRAMES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Database files
KNOWN_FACES_DB = DATABASE_DIR / "known_faces.json"
EMOTION_LOGS_DB = DATABASE_DIR / "emotion_logs.json"

# Camera settings
CAMERA_ID = 0
CAMERA_BACKEND = cv2.CAP_ANY

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20   # Reduced from 30 for lower CPU usage

# Face recognition settings
FACE_RECOGNITION_MODEL = "Facenet"
FACE_DETECTION_MODEL = "mtcnn"
SIMILARITY_THRESHOLD = 0.6
EMBEDDING_SIZE = 128

# Emotion detection settings
EMOTION_MODEL = "Emotion"
EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

# Frame capture settings
MAX_UNKNOWN_FRAMES = 30          # Reduced from 50
CAPTURE_EVERY_N_FRAMES = 3       # Better balance than 2

# Display settings
DISPLAY_SCALE = 1.0
FONT_SCALE = 0.8
FONT_THICKNESS = 2

TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 255, 0)
UNKNOWN_BOX_COLOR = (0, 0, 255)

# Robot reactions
ROBOT_REACTIONS = {
    "happy": {
        "tail_wag": "fast",
        "speed": "normal",
        "sound": "happy"
    },
    "sad": {
        "tail_wag": "slow",
        "speed": "slow",
        "sound": "sad"
    },
    "angry": {
        "tail_wag": "none",
        "speed": "cautious",
        "sound": "alert"
    },
    "fear": {
        "tail_wag": "none",
        "speed": "backward",
        "sound": "gentle"
    },
    "surprise": {
        "tail_wag": "fast",
        "speed": "stop",
        "sound": "surprised"
    },
    "neutral": {
        "tail_wag": "normal",
        "speed": "normal",
        "sound": "normal"
    }
}

# Voice settings
TTS_LANGUAGE = "en"
TTS_SPEED = 1.0

# Emotion logging
EMOTION_LOG_INTERVAL = 3   # Reduced from 5 for quicker logs