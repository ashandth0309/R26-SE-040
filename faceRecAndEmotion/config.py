"""
Configuration file for Smart AI Dog Robot
OPTIMIZED for smooth performance
"""

import os
import cv2
from pathlib import Path

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Base paths
BASE_DIR = Path(__file__).parent
DATABASE_DIR = BASE_DIR / "database"
UNKNOWN_FRAMES_DIR = BASE_DIR / "unknown_frames"
MODELS_DIR = BASE_DIR / "models"

# Create directories
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FRAMES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Database files
KNOWN_FACES_DB = DATABASE_DIR / "known_faces.json"
EMOTION_LOGS_DB = DATABASE_DIR / "emotion_logs.json"

# Camera settings - Standard for good performance
CAMERA_ID = 0
CAMERA_BACKEND = cv2.CAP_DSHOW
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Performance settings
FACE_DETECTION_INTERVAL = 3  # Run detection every 3 frames
EMOTION_DETECTION_INTERVAL = 6  # Run emotion every 6 frames
EMBEDDING_INTERVAL = 3  # Run embedding every 3 frames

# Display settings
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Colors
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 255, 0)
UNKNOWN_BOX_COLOR = (0, 0, 255)
EMOTION_COLORS = {
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "fear": (128, 0, 128),
    "surprise": (255, 255, 0),
    "disgust": (0, 255, 255),
    "neutral": (255, 255, 255)
}

# Face recognition
SIMILARITY_THRESHOLD = 0.6  # Lower threshold for better matching
EMBEDDING_SIZE = 128

# Emotion detection
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Frame management
MAX_UNKNOWN_FRAMES = 20
CAPTURE_EVERY_N_FRAMES = 5

# Robot reactions
ROBOT_REACTIONS = {
    "happy": {"tail_wag": "fast", "speed": "normal", "sound": "happy"},
    "sad": {"tail_wag": "slow", "speed": "slow", "sound": "sad"},
    "angry": {"tail_wag": "none", "speed": "cautious", "sound": "alert"},
    "fear": {"tail_wag": "none", "speed": "backward", "sound": "gentle"},
    "surprise": {"tail_wag": "fast", "speed": "stop", "sound": "surprised"},
    "neutral": {"tail_wag": "normal", "speed": "normal", "sound": "normal"}
}

# Voice settings
TTS_LANGUAGE = "en"
TTS_SPEED = 1.0

# Emotion logging
EMOTION_LOG_INTERVAL = 4