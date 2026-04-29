"""
Face Recognition Module using DeepFace and MTCNN
Fixed version for shape compatibility and Error Logging
"""

import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
import tensorflow as tf
from config import FACE_RECOGNITION_MODEL, EMBEDDING_SIZE
import time
import traceback  # Added for detailed error logging

class FaceRecognition:
    def __init__(self):
        print("🔄 Initializing Face Recognition Module...")
        
        # Initialize face detector
        self.detector = MTCNN()
        
        # Initialize face recognition model
        self.model_name = FACE_RECOGNITION_MODEL
        self.embedding_size = EMBEDDING_SIZE
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_size = 100
        
        # Minimum face size to process
        self.min_face_size = 50
        
        print(f"✅ Face Recognition Module initialized with {self.model_name}")
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using MTCNN
        Returns: list of (x, y, w, h) for each face
        """
        try:
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detector.detect_faces(rgb_frame)
            
            faces = []
            for det in detections:
                x, y, w, h = det['box']
                # Ensure coordinates are positive and within frame
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                # Filter out small faces
                if w > self.min_face_size and h > self.min_face_size:
                    faces.append((x, y, w, h))
            
            return faces
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def extract_face(self, frame, bbox):
        """
        Extract face region from frame and prepare for model input.
        DeepFace handles resizing and color conversion internally.
        """
        try:
            x, y, w, h = bbox
            
            # Extract face directly (No manual resize or RGB conversion needed here)
            face = frame[y:y+h, x:x+w]
            
            if face.size == 0:
                return None
            
            # Return the cropped face. DeepFace works best with raw BGR images.
            return face
            
        except Exception as e:
            print(f"Error extracting face: {e}")
            return None
    
    def get_embedding(self, face_img):
        """
        Extract face embedding using DeepFace with proper error handling
        and resizing to match model input requirements.
        """
        if face_img is None or face_img.size == 0:
            return None
        
        try:
            # === NEW CODE ADDED HERE ===
            # DeepFace Facenet model strictly expects 160x160 images
            # So we resize the extracted face here before passing it to AI
            resized_face = cv2.resize(face_img, (160, 160))
            
            # Convert BGR to RGB (DeepFace expects RGB)
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            # ===========================

            # Generate embedding using DeepFace
            embedding = DeepFace.represent(
                img_path=rgb_face, 
                model_name=self.model_name,
                enforce_detection=False,  # Don't enforce detection since we already have face
                detector_backend='skip'  # Skip detection
            )
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]["embedding"])
            else:
                return None
                
        except Exception as e:
            # This will print the FULL error message to help us debug
            print("\n❌ ---------------- EMBEDDING ERROR ---------------- ❌")
            traceback.print_exc()
            print("❌ ------------------------------------------------- ❌\n")
            return None
    
    def get_embeddings_batch(self, face_imgs):
        """
        Extract embeddings for multiple faces with error handling
        """
        embeddings = []
        for i, face in enumerate(face_imgs):
            try:
                emb = self.get_embedding(face)
                if emb is not None and len(emb) == self.embedding_size:
                    embeddings.append(emb)
                else:
                    print(f"Skipping face {i}: invalid embedding")
            except Exception as e:
                print(f"Error processing face {i}: {e}")
                continue
        
        return embeddings
    
    def extract_embeddings_from_frames(self, frames, bbox):
        """
        Extract embeddings from multiple frames of the same person
        Used during enrollment - more robust version
        """
        embeddings = []
        
        for i, frame in enumerate(frames):
            try:
                face = self.extract_face(frame, bbox)
                if face is not None:
                    emb = self.get_embedding(face)
                    if emb is not None:
                        embeddings.append(emb)
                        print(f"✅ Extracted embedding {len(embeddings)}/{len(frames)}")
                    else:
                        print(f"⚠️ Failed to get embedding from frame {i}")
                else:
                    print(f"⚠️ Failed to extract face from frame {i}")
            except Exception as e:
                print(f"⚠️ Error on frame {i}: {e}")
                continue
        
        print(f"📊 Successfully extracted {len(embeddings)} embeddings")
        return embeddings
    
    def validate_face_image(self, face_img):
        """
        Validate if face image is suitable for embedding extraction
        """
        if face_img is None:
            return False
        
        # Check shape
        if len(face_img.shape) != 3:
            return False
        
        if face_img.shape[2] != 3:
            return False
        
        # Check size
        h, w = face_img.shape[:2]
        if h < 50 or w < 50:
            return False
        
        # Check if image is too dark (optional)
        if np.mean(face_img) < 20:
            return False
        
        return True
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        print("🧹 Embedding cache cleared")