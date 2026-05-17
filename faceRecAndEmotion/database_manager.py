"""
Database Manager for Known Faces and Emotion Logs - FIXED VERSION
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
        person_id = hashlib.md5(f"{name}_{get_timestamp()}".encode()).hexdigest()[:8]
        
        person = {
            "id": person_id,
            "name": name,
            "embeddings": embeddings,
            "created_at": get_timestamp(),
            "total_frames": len(embeddings)
        }
        
        self.known_faces["persons"].append(person)
        save_json(KNOWN_FACES_DB, self.known_faces)
        print(f"✅ Person '{name}' added with ID: {person_id}")
        return person_id
    
    def find_person(self, embedding):
        """
        STRICTER MATCHING: Finds the person only if similarity is high enough.
        """
        if not self.known_faces["persons"]:
            return None, None
        
        best_match = None
        best_score = -1
        
        # We loop through every registered person
        for person in self.known_faces["persons"]:
            # Calculate the average similarity across all saved frames for this person
            # This is more stable than matching just ONE frame.
            person_similarities = []
            for known_embedding in person["embeddings"]:
                sim = cosine_similarity([embedding], [known_embedding])[0][0]
                person_similarities.append(sim)
            
            # Use the top 20% of matches to get a reliable average
            person_similarities.sort(reverse=True)
            top_mean_score = np.mean(person_similarities[:10]) 

            if top_mean_score > best_score:
                best_score = top_mean_score
                best_match = person["name"]
        
        # Check against the THRESHOLD in config.py
        # RECOMMENDATION: Set SIMILARITY_THRESHOLD = 0.75 or 0.8 in config.py
        if best_score >= SIMILARITY_THRESHOLD:
            return best_match, best_score
        else:
            # If the score is low, it returns None (UNKNOWN)
            return None, best_score

    def log_emotion(self, name, emotion, confidence):
        log_entry = {
            "timestamp": get_timestamp(),
            "name": name if name else "Unknown",
            "emotion": emotion,
            "confidence": confidence
        }
        self.emotion_logs["logs"].append(log_entry)
        if len(self.emotion_logs["logs"]) > 10000:
            self.emotion_logs["logs"] = self.emotion_logs["logs"][-10000:]
        save_json(EMOTION_LOGS_DB, self.emotion_logs)