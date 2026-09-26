import os
import json
import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

BASE_DIR = Path(__file__).parent
OBJECT_DB_PATH = BASE_DIR / "database" / "object_database.json"

class ObjectLearningModule:
    def __init__(self, threshold=0.70):
        self.threshold = threshold
        self.db_path = OBJECT_DB_PATH
        self.database = self.load_database()
        
        # Tracking & Interaction variables (main.py compatible)
        self.current_frame = None
        self.last_detected_name = "Unknown"
        self.is_learning = False
        self.learning_mode = False
        self.target_name = ""
        self.collected_features = []
        self.required_frames = 30
        
        print("⏳ Loading MobileNetV2 for Object Learning...")
        self.feature_extractor = MobileNetV2(
            weights="imagenet", 
            include_top=False, 
            pooling="avg", 
            input_shape=(224, 224, 3)
        )
        print("✅ Object Learning Module Ready!")

    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_database(self):
        os.makedirs(self.db_path.parent, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.database, f, indent=4)

    def extract_features(self, crop):
        img = cv2.resize(crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img, axis=0)
        img_preprocessed = preprocess_input(img_array)
        features = self.feature_extractor.predict(img_preprocessed, verbose=0)[0]
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features.tolist()

    def start_learning(self, obj_name):
        self.target_name = obj_name.strip().lower()
        self.collected_features = []
        self.is_learning = True
        self.learning_mode = True
        print(f"🎯 Started learning mode for: '{self.target_name}'")

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        
        # Periya porutkalukaaga box size 340-aaga maathappattulladhu
        box_size = 340
        bx1 = max(0, (w - box_size) // 2)
        by1 = max(0, (h - box_size) // 2)
        bx2 = min(w, bx1 + box_size)
        by2 = min(h, by1 + box_size)
        box = (bx1, by1, bx2, by2)

        crop = frame[by1:by2, bx1:bx2]
        status_text = ""

        # 1. Learning nadakkum podhu: Orange Box mattrum Frame Counter kaattum
        if self.is_learning and crop.size > 0:
            feat = self.extract_features(crop)
            self.collected_features.append(feat)
            status_text = f"Learning {self.target_name}: {len(self.collected_features)}/{self.required_frames}"
            
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 3)
            cv2.putText(frame, status_text, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            if len(self.collected_features) >= self.required_frames:
                if self.target_name not in self.database:
                    self.database[self.target_name] = []
                mean_feat = np.mean(self.collected_features, axis=0)
                norm = np.linalg.norm(mean_feat)
                if norm > 0:
                    mean_feat = mean_feat / norm
                self.database[self.target_name].append(mean_feat.tolist())
                self.save_database()
                print(f"🎉 Successfully learned object: {self.target_name}")
                self.is_learning = False
                self.learning_mode = False
                self.last_detected_name = self.target_name

        # 2. Database-la porul match aanaal mattum: Green Box mattrum match % kaattum
        elif crop.size > 0 and self.database:
            current_feat = np.array(self.extract_features(crop))
            best_match = "Unknown"
            best_score = 0.0

            for name, feat_list in self.database.items():
                for stored in feat_list:
                    score = float(np.dot(current_feat, np.array(stored)))
                    if score > best_score:
                        best_score = score
                        if score >= self.threshold:
                            best_match = name

            self.last_detected_name = best_match
            if best_match != "Unknown":
                status_text = f"{best_match.upper()} ({int(best_score * 100)}%)"
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, status_text, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. Normal nerathil verum box edhuvum thiraiyil varaadhu

        return frame, box, status_text, self.last_detected_name