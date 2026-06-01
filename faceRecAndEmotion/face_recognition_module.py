"""
Face Recognition Module - HYBRID VERSION

This module supports BOTH:

1. TensorFlow trained model recognition
   - Uses models/face_model.keras
   - Uses models/class_indices.json
   - Good for trained dataset persons like Sobiya, Ashandth

2. Old enrollment/database recognition
   - Uses get_embedding()
   - Used by DatabaseManager
   - Allows pressing 'e' in main.py to enroll new people
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path

try:
    import tensorflow as tf
except Exception:
    tf = None

from faceRecAndEmotion.config import MODELS_DIR


class FaceRecognition:
    def __init__(self):
        print("🔄 Initializing Hybrid Face Recognition Module...")

        # ======================
        # Face detector
        # ======================
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            print("❌ Failed to load face cascade!")
        else:
            print("✅ Face cascade loaded successfully")

        self.min_face_size = (80, 80)
        self.scale_factor = 1.2
        self.min_neighbors = 6

        # Cache for performance
        self.last_detection_time = 0
        self.detection_interval = 0.05
        self.cached_faces = []

        # ======================
        # TensorFlow trained model settings
        # ======================
        self.model = None
        self.class_names = {}
        self.img_size = (160, 160)

        # Strict unknown logic
        # If strangers show as Sobiya/Ashandth, increase these.
        self.model_confidence_threshold = 0.95
        self.model_margin_threshold = 0.40

        model_path = Path(MODELS_DIR) / "face_model.keras"
        class_map_path = Path(MODELS_DIR) / "class_indices.json"

        if tf is None:
            print("⚠️ TensorFlow not installed. Trained model disabled.")
        elif model_path.exists() and class_map_path.exists():
            try:
                self.model = tf.keras.models.load_model(str(model_path))

                with open(class_map_path, "r", encoding="utf-8") as f:
                    class_indices = json.load(f)

                self.class_names = {int(v): k for k, v in class_indices.items()}

                print(f"✅ Trained face model loaded: {model_path}")
                print(f"✅ Class mapping loaded: {self.class_names}")

            except Exception as e:
                print(f"⚠️ Failed to load trained model: {e}")
                self.model = None
        else:
            print("⚠️ Trained model files not found.")
            print(f"   Expected: {model_path}")
            print(f"   Expected: {class_map_path}")
            print("   Old enrollment/database system will still work.")

        # ======================
        # Old database recognition threshold
        # ======================
        self.match_threshold = 0.88

        print("✅ Hybrid Face Recognition Module Ready")

    # =====================================================
    # FACE DETECTION
    # =====================================================
    def detect_faces(self, frame):
        """
        Detect faces in frame using OpenCV Haar Cascade.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if frame is None or frame.size == 0:
            return []

        current_time = time.time()

        if current_time - self.last_detection_time < self.detection_interval:
            return self.cached_faces

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size
            )

            detected_faces = [
                (int(x), int(y), int(w), int(h))
                for (x, y, w, h) in faces
            ]

            self.cached_faces = detected_faces
            self.last_detection_time = current_time

            return detected_faces

        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def extract_face(self, frame, bbox):
        """
        Extract face ROI from frame with padding.
        """
        try:
            x, y, w, h = bbox

            padding = int(0.20 * w)

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            face = frame[y1:y2, x1:x2]

            if face is None or face.size == 0:
                return None

            return face

        except Exception:
            return None

    def validate_face(self, face):
        """
        Validate if face image is good quality.
        """
        if face is None or face.size == 0:
            return False

        h, w = face.shape[:2]

        if h < 60 or w < 60:
            return False

        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face

            brightness = np.mean(gray)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if brightness < 30 or brightness > 235:
                return False

            if blur_score < 15:
                return False

        except Exception:
            return False

        return True

    # =====================================================
    # TRAINED TENSORFLOW MODEL RECOGNITION
    # =====================================================
    def preprocess_for_model(self, face_img):
        """
        Preprocess face exactly like training:
        RGB, 160x160, rescale 1/255.
        """
        face = cv2.resize(face_img, self.img_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        return face

    def predict_person(self, face_img):
        """
        Predict using trained TensorFlow model.

        Returns:
            (name, confidence_percent, source)

        If not confident:
            (None, confidence_percent, "model_unknown")
        """
        if self.model is None:
            return None, 0.0, "model_not_loaded"

        if face_img is None:
            return None, 0.0, "no_face"

        try:
            face = self.preprocess_for_model(face_img)
            pred = self.model.predict(face, verbose=0)[0]

            index = int(np.argmax(pred))
            confidence = float(pred[index])

            sorted_preds = np.sort(pred)
            second_best = float(sorted_preds[-2]) if len(sorted_preds) > 1 else 0.0
            margin = confidence - second_best

            predicted_name = self.class_names.get(index, None)

            print(
                f"🤖 MODEL: {predicted_name} "
                f"conf={confidence:.3f} "
                f"margin={margin:.3f} "
                f"all={pred}"
            )

            if (
                predicted_name is not None
                and confidence >= self.model_confidence_threshold
                and margin >= self.model_margin_threshold
            ):
                return predicted_name, confidence * 100, "trained_model"

            return None, confidence * 100, "model_unknown"

        except Exception as e:
            print(f"Face model prediction error: {e}")
            return None, 0.0, "model_error"

    # =====================================================
    # OLD ENROLLMENT / DATABASE EMBEDDING SYSTEM
    # =====================================================
    def get_embedding(self, face_img):
        """
        Generate simple face embedding.
        This is kept for old enrollment system.
        Press 'e' enrollment in main.py still depends on this.
        """
        if face_img is None:
            return None

        try:
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img

            gray = cv2.resize(gray, (128, 128))
            gray = cv2.equalizeHist(gray)

            embedding = []

            # Basic intensity features
            embedding.append(np.mean(gray))
            embedding.append(np.std(gray))

            # Grid features
            h, w = gray.shape
            grid_size = 8
            cell_h = h // grid_size
            cell_w = w // grid_size

            for i in range(grid_size):
                for j in range(grid_size):
                    y1 = i * cell_h
                    y2 = (i + 1) * cell_h
                    x1 = j * cell_w
                    x2 = (j + 1) * cell_w

                    cell = gray[y1:y2, x1:x2]
                    embedding.append(np.mean(cell))
                    embedding.append(np.std(cell))

            # Histogram features
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-8)
            embedding.extend(hist.tolist())

            # Edge features
            edges = cv2.Canny(gray, 80, 160)
            embedding.append(np.mean(edges))
            embedding.append(np.std(edges))

            embedding = np.array(embedding, dtype=np.float32)

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def get_embeddings_batch(self, faces):
        """
        Generate embeddings for multiple faces.
        Used by enrollment in main.py.
        """
        if not faces:
            return []

        embeddings = []

        for face in faces:
            emb = self.get_embedding(face)
            if emb is not None:
                embeddings.append(emb)

        return embeddings

    def compare_faces(self, embedding1, embedding2):
        """
        Compare two old-style embeddings using cosine similarity.
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        embedding1 = np.asarray(embedding1, dtype=np.float32)
        embedding2 = np.asarray(embedding2, dtype=np.float32)

        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
        else:
            similarity = 0.0

        return float(similarity)