"""
Database Manager - Working version
"""

import numpy as np
from faceRecAndEmotion.utils.helpers import load_json, save_json, get_timestamp
from faceRecAndEmotion.config import KNOWN_FACES_DB, EMOTION_LOGS_DB, SIMILARITY_THRESHOLD
import hashlib
import os

class DatabaseManager:
    def __init__(self):
        self.known_faces = load_json(KNOWN_FACES_DB, {"persons": []})
        self.emotion_logs = load_json(EMOTION_LOGS_DB, {"logs": []})
        
        # Print registered users
        print("\n📋 Registered users:")
        for p in self.known_faces.get("persons", []):
            emb_count = len(p.get("embeddings", []))
            print(f"   ✅ {p['name']} ({emb_count} samples)")
        print("")
    
    def add_person(self, name, embeddings):
        """Add a new person to database cleanly without float32 serialization issues"""
        if embeddings is None or len(embeddings) == 0:
            print(f"❌ No valid embeddings for {name}")
            return None
        
        person_id = hashlib.md5(f"{name}_{get_timestamp()}".encode()).hexdigest()[:8]
        
        # NumPy array or Nested list handling
        arr = np.array(embeddings, dtype=np.float32)
        
        # 1D array ஆக வந்தால் (Single mean embedding)
        if arr.ndim == 1:
            processed = [arr.astype(float).tolist()]
            avg_embedding = arr.astype(float).tolist()
        else:
            # Multi-frame embeddings ஆக வந்தால்
            processed = [row.astype(float).tolist() for row in arr]
            avg_embedding = np.mean(arr, axis=0).astype(float).tolist()
        
        person = {
            "id": person_id,
            "name": name,
            "embeddings": processed,
            "avg_embedding": avg_embedding,
            "created_at": get_timestamp(),
            "total_frames": len(processed)
        }
        
        if "persons" not in self.known_faces:
            self.known_faces["persons"] = []
            
        self.known_faces["persons"].append(person)
        save_json(KNOWN_FACES_DB, self.known_faces)
        print(f"✅ Enrolled '{name}' with {len(processed)} face samples")
        return person_id
    
    def find_person(self, embedding):
        """
        Find person by face embedding
        Returns (name, similarity) or (None, best_score)
        """
        if not self.known_faces.get("persons") or embedding is None:
            return None, 0.0
        
        embedding = np.array(embedding, dtype=np.float32).flatten()
        
        best_match = None
        best_score = -1.0
        
        for person in self.known_faces["persons"]:
            if "avg_embedding" in person and person["avg_embedding"]:
                avg_emb = np.array(person["avg_embedding"], dtype=np.float32).flatten()
                
                # Dimension சரிபார்த்தல்
                if len(embedding) != len(avg_emb):
                    continue
                
                # Cosine similarity
                norm1 = np.linalg.norm(embedding)
                norm2 = np.linalg.norm(avg_emb)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = float(np.dot(embedding, avg_emb) / (norm1 * norm2))
                else:
                    similarity = 0.0
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = person["name"]
        
        if best_score >= SIMILARITY_THRESHOLD:
            return best_match, best_score
        return None, best_score
    
    def log_emotion(self, name, emotion, confidence):
        """Log emotion detection"""
        if "logs" not in self.emotion_logs:
            self.emotion_logs["logs"] = []
            
        self.emotion_logs["logs"].append({
            "timestamp": get_timestamp(),
            "name": name or "Unknown",
            "emotion": emotion,
            "confidence": float(confidence)
        })
        
        # Keep only last 5000 logs
        if len(self.emotion_logs["logs"]) > 5000:
            self.emotion_logs["logs"] = self.emotion_logs["logs"][-5000:]
        
        save_json(EMOTION_LOGS_DB, self.emotion_logs)