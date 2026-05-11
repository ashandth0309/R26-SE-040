"""
Emotion Detection Module using Custom Trained Model (FER2013)
Optimized for Speed and Accuracy
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import EMOTIONS
import time
import os

# Suppress unnecessary TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EmotionDetector:
    def __init__(self, model_path='models/emotion_model.h5'):
        print("🔄 Initializing Custom Emotion Detection Module...")
        
        # 1. Load your trained model
        try:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                print(f"✅ Custom Model loaded from {model_path}")
            else:
                print(f"❌ Model file not found at {model_path}! Please run the training script first.")
                self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.last_emotion = None
        self.emotion_history = []
        self.history_size = 5 
        
        # Throttling (To control the processing speed)
        self.last_detection_time = 0
        self.detection_interval = 0.3  # Check once every 0.3 seconds

    def detect_emotion(self, face_img):
        """
        Detect emotion using the FER2013 trained model
        """
        current_time = time.time()
        
        # Skip detection if called too frequently (Throttling)
        if current_time - self.last_detection_time < self.detection_interval:
            return self.last_emotion or "neutral", {}

        if self.model is None:
            return "neutral", {}

        try:
            # Step 1: Convert image to Grayscale (FER2013 standard)
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Step 2: Resize to 48x48 pixels
            resized_face = cv2.resize(gray_face, (48, 48))
            
            # Step 3: Normalization (Scale pixels to 0-1 range)
            normalized_face = resized_face / 255.0
            
            # Step 4: Reshape for the model input (Batch, Width, Height, Channels)
            reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

            # Step 5: Prediction
            predictions = self.model.predict(reshaped_face, verbose=0)[0]
            max_index = np.argmax(predictions)
            dominant_emotion = self.emotions[max_index]

            # Smoothing (To avoid flickering between emotions)
            self.emotion_history.append(dominant_emotion)
            if len(self.emotion_history) > self.history_size:
                self.emotion_history.pop(0)
            
            from collections import Counter
            smoothed_emotion = Counter(self.emotion_history).most_common(1)[0][0]

            self.last_emotion = smoothed_emotion
            self.last_detection_time = current_time

            # Probability dictionary output
            emotions_dict = {self.emotions[i]: float(predictions[i]) for i in range(len(self.emotions))}
            
            return smoothed_emotion, emotions_dict

        except Exception as e:
            print(f"⚠️ Detection Error: {e}")
            return "neutral", {}

    def get_emotion_color(self, emotion):
        """Return BGR color code for specific emotions"""
        colors = {
            "happy": (0, 255, 0), "sad": (255, 0, 0), "angry": (0, 0, 255),
            "fear": (128, 0, 128), "surprise": (255, 255, 0),
            "disgust": (0, 255, 255), "neutral": (255, 255, 255)
        }
        return colors.get(emotion, (255, 255, 255))

    def get_emotion_emoji(self, emotion):
        """Return emoji string for specific emotions"""
        emojis = {
            "happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
            "surprise": "😲", "disgust": "🤢", "neutral": "😐"
        }
        return emojis.get(emotion, "❓")