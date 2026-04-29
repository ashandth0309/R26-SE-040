"""
Emotion Detection Module using DeepFace
Fixed version with reduced logging
"""

import cv2
import numpy as np
from deepface import DeepFace
from config import EMOTIONS
import time
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EmotionDetector:
    def __init__(self):
        print("🔄 Initializing Emotion Detection Module...")
        
        self.emotions = EMOTIONS
        self.last_emotion = None
        self.last_confidence = 0
        self.emotion_history = []
        self.history_size = 5  # Reduced for faster response
        
        # Throttling
        self.last_detection_time = 0
        self.detection_interval = 0.5  # Only detect every 0.5 seconds
        
        print(f"✅ Emotion Detector initialized with {len(self.emotions)} emotions")
    
    def detect_emotion(self, face_img):
        """
        Detect emotion from face image with throttling
        """
        current_time = time.time()
        
        # Throttle detection to avoid too many calls
        if current_time - self.last_detection_time < self.detection_interval:
            return self.last_emotion or "neutral", {
                emotion: 0 for emotion in self.emotions
            }
        
        try:
            # Convert BGR to RGB if needed
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                # Assume it's BGR, convert to RGB
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                rgb_face = face_img
            
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(
                img_path=rgb_face,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True  # Suppress output
            )
            
            if isinstance(result, list):
                result = result[0]
            
            # Get emotion probabilities
            emotions_dict = result['emotion']
            
            # Find dominant emotion
            dominant_emotion = max(emotions_dict, key=emotions_dict.get)
            
            # Update history
            self.emotion_history.append(dominant_emotion)
            if len(self.emotion_history) > self.history_size:
                self.emotion_history.pop(0)
            
            # Smooth emotion (take most common in history)
            if len(self.emotion_history) >= 3:
                from collections import Counter
                emotion_counts = Counter(self.emotion_history)
                dominant_emotion = emotion_counts.most_common(1)[0][0]
            
            self.last_emotion = dominant_emotion
            self.last_detection_time = current_time
            
            return dominant_emotion, emotions_dict
            
        except Exception as e:
            # Silent fail - just return neutral
            self.last_detection_time = current_time
            return "neutral", {emotion: 0 for emotion in self.emotions}
    
    def get_emotion_color(self, emotion):
        """Get color associated with emotion for display"""
        colors = {
            "happy": (0, 255, 0),
            "sad": (255, 0, 0),
            "angry": (0, 0, 255),
            "fear": (128, 0, 128),
            "surprise": (255, 255, 0),
            "disgust": (0, 255, 255),
            "neutral": (255, 255, 255)
        }
        return colors.get(emotion, (255, 255, 255))
    
    def get_emotion_emoji(self, emotion):
        """Get emoji for emotion"""
        emojis = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "fear": "😨",
            "surprise": "😲",
            "disgust": "🤢",
            "neutral": "😐"
        }
        return emojis.get(emotion, "❓")