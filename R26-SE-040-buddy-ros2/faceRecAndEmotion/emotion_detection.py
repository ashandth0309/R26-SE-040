"""
Safe Emotion Detection Module

Uses DeepFace if available.
If DeepFace fails, falls back to neutral emotion so the robot still runs.
"""

from faceRecAndEmotion.config import EMOTION_COLORS, EMOTIONS


class EmotionDetector:
    def __init__(self):
        print("🔄 Initializing Emotion Detection Module...")

        self.deepface_available = False
        self.DeepFace = None

        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.deepface_available = True
            print("✅ DeepFace emotion detector loaded")
        except Exception as error:
            print(f"⚠️ DeepFace unavailable: {error}")
            print("⚠️ Emotion detection will use neutral fallback")
            self.deepface_available = False

        print("✅ Emotion Detection Module Ready")

    def detect_emotion(self, face_img):
        """
        Returns:
            emotion, confidence, all_emotions
        """
        if face_img is None:
            return "neutral", 0.0, {"neutral": 100.0}

        if not self.deepface_available:
            return "neutral", 100.0, {"neutral": 100.0}

        try:
            result = self.DeepFace.analyze(
                face_img,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            emotion_scores = result.get("emotion", {})
            dominant_emotion = result.get("dominant_emotion", "neutral")

            if not emotion_scores:
                return "neutral", 100.0, {"neutral": 100.0}

            confidence = float(emotion_scores.get(dominant_emotion, 0.0))

            fixed_scores = {}
            for emotion in EMOTIONS:
                fixed_scores[emotion] = float(emotion_scores.get(emotion, 0.0))

            return dominant_emotion, confidence, fixed_scores

        except Exception as error:
            print(f"⚠️ Emotion detection failed: {error}")
            return "neutral", 100.0, {"neutral": 100.0}

    def get_emotion_color(self, emotion):
        return EMOTION_COLORS.get(emotion, EMOTION_COLORS.get("neutral", (255, 255, 255)))

    def get_emotion_emoji(self, emotion):
        emojis = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "fear": "😨",
            "surprise": "😲",
            "disgust": "🤢",
            "neutral": "😐",
        }

        return emojis.get(emotion, "😐")