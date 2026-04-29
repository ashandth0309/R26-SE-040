"""
Database Manager for Known Faces and Emotion Logs
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import load_json, save_json, get_timestamp
from config import KNOWN_FACES_DB, EMOTION_LOGS_DB, SIMILARITY_THRESHOLD
import hashlib

class DatabaseManager:
    def __init__(self):
        self.known_faces = load_json(KNOWN_FACES_DB, {"persons": []})
        self.emotion_logs = load_json(EMOTION_LOGS_DB, {"logs": []})
        
    def add_person(self, name, embeddings):
        """
        Add a new person with their face embeddings
        embeddings: list of 128-d face embeddings
        """
        person_id = hashlib.md5(f"{name}_{get_timestamp()}".encode()).hexdigest()[:8]
        
        person = {
            "id": person_id,
            "name": name,
            "embeddings": embeddings,  # Store multiple embeddings for robustness
            "created_at": get_timestamp(),
            "total_frames": len(embeddings)
        }
        
        self.known_faces["persons"].append(person)
        save_json(KNOWN_FACES_DB, self.known_faces)
        print(f"✅ Person '{name}' added with ID: {person_id}")
        return person_id
    
    def find_person(self, embedding):
        """
        Find a person by embedding similarity
        Returns: (name, similarity_score) or (None, None)
        """
        if not self.known_faces["persons"]:
            return None, None
        
        best_match = None
        best_score = -1
        
        for person in self.known_faces["persons"]:
            # Compare with all embeddings of this person
            for known_embedding in person["embeddings"]:
                similarity = cosine_similarity([embedding], [known_embedding])[0][0]
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = person["name"]
        
        # Check if score meets threshold
        if best_score >= SIMILARITY_THRESHOLD:
            return best_match, best_score
        else:
            return None, None
    
    def log_emotion(self, name, emotion, confidence):
        """Log emotion detection event"""
        log_entry = {
            "timestamp": get_timestamp(),
            "name": name,
            "emotion": emotion,
            "confidence": confidence
        }
        
        self.emotion_logs["logs"].append(log_entry)
        
        # Keep only last 10000 logs to prevent file from growing too large
        if len(self.emotion_logs["logs"]) > 10000:
            self.emotion_logs["logs"] = self.emotion_logs["logs"][-10000:]
        
        save_json(EMOTION_LOGS_DB, self.emotion_logs)
        
    def get_emotion_stats(self, name=None, days=None):
        """Get emotion statistics"""
        import pandas as pd
        from datetime import datetime, timedelta
        
        if not self.emotion_logs["logs"]:
            return None
        
        df = pd.DataFrame(self.emotion_logs["logs"])
        
        if name:
            df = df[df["name"] == name]
        
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            df = df[pd.to_datetime(df["timestamp"]) > cutoff]
        
        if len(df) == 0:
            return None
        
        stats = {
            "total_entries": len(df),
            "emotion_counts": df["emotion"].value_counts().to_dict(),
            "avg_confidence": df["confidence"].mean(),
            "most_common_emotion": df["emotion"].mode()[0] if len(df) > 0 else None
        }
        
        return stats
    
    def get_all_persons(self):
        """Get list of all registered persons"""
        return [{"id": p["id"], "name": p["name"], "created_at": p["created_at"]} 
                for p in self.known_faces["persons"]]
    
    def delete_person(self, person_id):
        """Delete a person from database"""
        self.known_faces["persons"] = [
            p for p in self.known_faces["persons"] 
            if p["id"] != person_id
        ]
        save_json(KNOWN_FACES_DB, self.known_faces)
        print(f"Deleted person with ID: {person_id}")