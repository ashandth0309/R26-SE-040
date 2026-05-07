"""
Fixed Face Recognition Module (DeepFace + MTCNN)
- Fixes shape mismatch error (160,160,3)
- Ensures single image input
- Safer cropping
- Stable embedding generation
"""

import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
import traceback

from config import FACE_RECOGNITION_MODEL, EMBEDDING_SIZE


class FaceRecognition:
    def __init__(self):
        print("🔄 Initializing Face Recognition Module...")

        self.detector = MTCNN()

        self.model_name = FACE_RECOGNITION_MODEL
        self.embedding_size = EMBEDDING_SIZE

        self.min_face_size = 50

        print(f"✅ Model loaded: {self.model_name}")

    # ---------------- FACE DETECTION ----------------
    def detect_faces(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(rgb)

            faces = []

            for r in results:
                x, y, w, h = r['box']

                if w < self.min_face_size or h < self.min_face_size:
                    continue

                x = max(0, x)
                y = max(0, y)

                faces.append((x, y, w, h))

            return faces

        except Exception as e:
            print("Face detection error:", e)
            return []

    # ---------------- FACE EXTRACTION ----------------
    def extract_face(self, frame, bbox):
        try:
            x, y, w, h = bbox

            face = frame[y:y+h, x:x+w]

            if face is None or face.size == 0:
                return None

            # FORCE correct shape (IMPORTANT FIX)
            face = cv2.resize(face, (160, 160))

            # Ensure 3 channels
            if len(face.shape) == 2:
                face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)

            if face.shape[-1] != 3:
                return None

            return face

        except Exception as e:
            print("Extract face error:", e)
            return None

    # ---------------- EMBEDDING ----------------
    def get_embedding(self, face_img):
        if face_img is None:
            return None

        try:
            # FINAL SAFETY CHECK (VERY IMPORTANT)
            if face_img.shape != (160, 160, 3):
                face_img = cv2.resize(face_img, (160, 160))

            # Convert to RGB (DeepFace requirement)
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # ADD BATCH DIMENSION CORRECTLY (FIXES YOUR ERROR)
            rgb_img = np.expand_dims(rgb_img, axis=0)

            embedding = DeepFace.represent(
                img_path=rgb_img,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend="skip"
            )

            return np.array(embedding[0]["embedding"])

        except Exception:
            print("\n❌ EMBEDDING ERROR ❌")
            traceback.print_exc()
            return None

    # ---------------- BATCH ----------------
    def get_embeddings_batch(self, faces):
        embeddings = []

        for i, f in enumerate(faces):
            emb = self.get_embedding(f)
            if emb is not None:
                embeddings.append(emb)

        return embeddings

    # ---------------- VALIDATION ----------------
    def validate_face(self, face):
        if face is None:
            return False

        if face.shape[0] < 50 or face.shape[1] < 50:
            return False

        if len(face.shape) != 3:
            return False

        return True