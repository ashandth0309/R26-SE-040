"""
Emotion Detection Module - Working with DeepFace + Percentage Display
"""

import os
import cv2
import numpy as np
import time
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EmotionDetector:
    def __init__(self):
        print("🔄 Initializing Emotion Detection Module...")
        
        self.emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]
        self.last_emotion = "neutral"
        self.last_confidence = 0.0
        self.last_probabilities = {emotion: 0.0 for emotion in self.emotions}
        
        # Smoothing history
        self.emotion_history = []
        self.confidence_history = []
        self.history_size = 5
        
        # Timing
        self.last_detection_time = 0
        self.detection_interval = 0.3  # 300ms between detections
        
        # Try to load DeepFace, fallback to simple method
        self.use_deepface = False
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.use_deepface = True
            print("✅ Using DeepFace for emotion detection")
        except ImportError:
            print("⚠️ DeepFace not available, using simple detector")
        
        print("✅ Emotion Detector Ready")
    
    def detect_emotion(self, face_img):
        """
        Detect emotion from face image
        Returns: (emotion, confidence_percentage, all_emotions_with_percentages)
        """
        if face_img is None or face_img.size == 0:
            return self.last_emotion, self.last_confidence, self.last_probabilities
        
        current_time = time.time()
        
        # Return cached result if within interval
        if current_time - self.last_detection_time < self.detection_interval:
            return self.last_emotion, self.last_confidence, self.last_probabilities
        
        try:
            if self.use_deepface:
                # Use DeepFace for accurate detection
                result = self.DeepFace.analyze(
                    img_path=face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True
                )
                
                if isinstance(result, list):
                    result = result[0]
                
                emotions_dict = result['emotion']
                dominant_emotion = max(emotions_dict, key=emotions_dict.get)
                confidence = emotions_dict[dominant_emotion]  # This is already percentage (0-100)
                
                # Get all emotions with percentages
                all_emotions = emotions_dict.copy()
                
            else:
                # Simple fallback detector
                dominant_emotion, confidence = self._simple_emotion_detector(face_img)
                confidence = confidence * 100  # Convert to percentage
                all_emotions = {e: 0 for e in self.emotions}
                all_emotions[dominant_emotion] = confidence
            
            # Smooth with history
            self.emotion_history.append(dominant_emotion)
            self.confidence_history.append(confidence)
            
            if len(self.emotion_history) > self.history_size:
                self.emotion_history.pop(0)
                self.confidence_history.pop(0)
            
            # Get smoothed emotion (most common in history)
            if len(self.emotion_history) >= 3:
                emotion_counts = Counter(self.emotion_history)
                dominant_emotion = emotion_counts.most_common(1)[0][0]
                
                # Average confidence for smoothed emotion
                avg_confidence = np.mean([c for e, c in zip(self.emotion_history, self.confidence_history) 
                                         if e == dominant_emotion])
                confidence = avg_confidence
            
            # Update probabilities dictionary with percentages
            self.last_probabilities = all_emotions
            
            self.last_emotion = dominant_emotion
            self.last_confidence = confidence
            self.last_detection_time = current_time
            
            return dominant_emotion, confidence, all_emotions
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            self.last_detection_time = current_time
            return self.last_emotion, self.last_confidence, self.last_probabilities
    
    def _simple_emotion_detector(self, face_img):
        """Simple fallback emotion detector"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Simple metrics
            mean_brightness = np.mean(gray)
            
            # Detect mouth region (bottom part)
            h, w = gray.shape
            mouth_region = gray[int(h*0.6):, :]
            mouth_mean = np.mean(mouth_region) if mouth_region.size > 0 else mean_brightness
            
            # Simple heuristic with confidence
            diff = mouth_mean - mean_brightness
            
            if diff > 15:
                confidence = min(90, 60 + diff)
                return "happy", confidence / 100
            elif diff < -15:
                confidence = min(90, 60 - diff)
                return "sad", confidence / 100
            else:
                return "neutral", 0.5
                
        except:
            return "neutral", 0.5
    
    def get_emotion_percentage_text(self, emotion, confidence):
        """Get formatted percentage text for display"""
        return f"{emotion.upper()}: {confidence:.1f}%"
    
    def get_all_emotions_text(self, emotions_dict):
        """Get formatted string of all emotions with percentages"""
        if not emotions_dict:
            return ""
        
        # Sort by confidence (highest first)
        sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Format top 3 emotions
        lines = []
        for emotion, conf in sorted_emotions[:3]:
            if conf > 0:
                emoji = self.get_emotion_emoji(emotion)
                lines.append(f"{emoji} {emotion}: {conf:.1f}%")
        
        return " | ".join(lines)
    
    def get_emotion_color(self, emotion):
        """Get color associated with emotion"""
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
    
    def get_confidence_bar(self, confidence, width=20):
        """Create a text-based confidence bar"""
        filled = int(width * confidence / 100)
        bar = "█" * filled + "░" * (width - filled)
        return bar