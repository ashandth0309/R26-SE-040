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
        for p in self.known_faces["persons"]:
            emb_count = len(p.get("embeddings", []))
            print(f"   ✅ {p['name']} ({emb_count} samples)")
        print("")
    
    def add_person(self, name, embeddings):
        """Add a new person to database"""
        if not embeddings:
            print(f"❌ No valid embeddings for {name}")
            return None
        
        person_id = hashlib.md5(f"{name}_{get_timestamp()}".encode()).hexdigest()[:8]
        
        # Convert embeddings to list for JSON serialization
        processed = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                processed.append(emb.tolist())
            else:
                processed.append(emb)
        
        # Compute average embedding
        avg_embedding = np.mean([np.array(emb) for emb in processed], axis=0).tolist()
        
        person = {
            "id": person_id,
            "name": name,
            "embeddings": processed,
            "avg_embedding": avg_embedding,
            "created_at": get_timestamp(),
            "total_frames": len(processed)
        }
        
        self.known_faces["persons"].append(person)
        save_json(KNOWN_FACES_DB, self.known_faces)
        print(f"✅ Enrolled '{name}' with {len(processed)} face samples")
        return person_id
    
    def find_person(self, embedding):
        """
        Find person by face embedding
        Returns (name, similarity) or (None, best_score)
        """
        if not self.known_faces["persons"] or embedding is None:
            return None, 0.0
        
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        best_match = None
        best_score = -1
        
        for person in self.known_faces["persons"]:
            if "avg_embedding" in person:
                avg_emb = np.array(person["avg_embedding"])
                
                # Check dimension match
                if len(embedding) != len(avg_emb):
                    continue
                
                # Cosine similarity
                dot_product = np.dot(embedding, avg_emb)
                norm1 = np.linalg.norm(embedding)
                norm2 = np.linalg.norm(avg_emb)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                else:
                    similarity = 0
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = person["name"]
        
        if best_score >= SIMILARITY_THRESHOLD:
            return best_match, best_score
        return None, best_score
    
    def log_emotion(self, name, emotion, confidence):
        """Log emotion detection"""
        self.emotion_logs["logs"].append({
            "timestamp": get_timestamp(),
            "name": name or "Unknown",
            "emotion": emotion,
            "confidence": confidence
        })
        
        # Keep only last 5000 logs
        if len(self.emotion_logs["logs"]) > 5000:
            self.emotion_logs["logs"] = self.emotion_logs["logs"][-5000:]
        
        save_json(EMOTION_LOGS_DB, self.emotion_logs)