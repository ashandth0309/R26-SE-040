"""
Simple OpenCV LBPH face memory.

This works without face_model.keras.
It lets Buddy remember faces captured through voice:
    "this is me my name is Ashandth"
"""

import os
import json
import cv2
import numpy as np

from faceRecAndEmotion.config import DATABASE_DIR


class SimpleFaceMemory:
    def __init__(self):
        self.base_dir = os.path.join(str(DATABASE_DIR), "lbph_faces")
        self.model_path = os.path.join(str(DATABASE_DIR), "lbph_model.yml")
        self.labels_path = os.path.join(str(DATABASE_DIR), "lbph_labels.json")

        os.makedirs(self.base_dir, exist_ok=True)

        self.labels = self._load_labels()
        self.recognizer = self._create_recognizer()

        if os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
                print("✅ LBPH face memory loaded")
            except Exception as error:
                print(f"⚠️ Could not load LBPH model: {error}")

    def _create_recognizer(self):
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "cv2.face is missing. Install opencv-contrib-python: "
                "pip uninstall opencv-python -y && pip install opencv-contrib-python"
            )

        return cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )

    def _load_labels(self):
        if not os.path.exists(self.labels_path):
            return {}

        try:
            with open(self.labels_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {}

    def _save_labels(self):
        with open(self.labels_path, "w", encoding="utf-8") as file:
            json.dump(self.labels, file, indent=2, ensure_ascii=False)

    def _next_label_id(self):
        if not self.labels:
            return 1

        return max(int(label_id) for label_id in self.labels.keys()) + 1

    def _get_or_create_label(self, name):
        for label_id, label_name in self.labels.items():
            if label_name.lower() == name.lower():
                return int(label_id)

        label_id = self._next_label_id()
        self.labels[str(label_id)] = name
        self._save_labels()
        return label_id

    def _prepare_face(self, face_img):
        if face_img is None:
            return None

        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        gray = cv2.resize(gray, (160, 160))
        gray = cv2.equalizeHist(gray)

        return gray

    def add_person(self, name, face_images):
        label_id = self._get_or_create_label(name)

        person_dir = os.path.join(self.base_dir, name.replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)

        existing_count = len([
            file for file in os.listdir(person_dir)
            if file.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        saved = 0

        for index, face_img in enumerate(face_images):
            prepared = self._prepare_face(face_img)

            if prepared is None:
                continue

            file_path = os.path.join(
                person_dir,
                f"{existing_count + index + 1:04d}.jpg"
            )

            cv2.imwrite(file_path, prepared)
            saved += 1

        self.train()

        return saved

    def train(self):
        faces = []
        ids = []

        for label_id, name in self.labels.items():
            person_dir = os.path.join(self.base_dir, name.replace(" ", "_"))

            if not os.path.exists(person_dir):
                continue

            for file_name in os.listdir(person_dir):
                if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                path = os.path.join(person_dir, file_name)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img = cv2.resize(img, (160, 160))

                faces.append(img)
                ids.append(int(label_id))

        if not faces:
            print("⚠️ No LBPH faces to train")
            return False

        self.recognizer.train(faces, np.array(ids))
        self.recognizer.save(self.model_path)
        self._save_labels()

        print(f"✅ LBPH trained with {len(faces)} face images")
        return True

    def predict(self, face_img):
        if not os.path.exists(self.model_path):
            return None, 0.0

        prepared = self._prepare_face(face_img)

        if prepared is None:
            return None, 0.0

        try:
            label_id, distance = self.recognizer.predict(prepared)
        except Exception as error:
            print(f"⚠️ LBPH predict failed: {error}")
            return None, 0.0

        # LBPH lower distance means better match.
        # This rough confidence conversion is enough for demo.
        confidence = max(0.0, min(100.0, 100.0 - distance))

        if distance > 75:
            return None, confidence

        name = self.labels.get(str(label_id))

        return name, confidence