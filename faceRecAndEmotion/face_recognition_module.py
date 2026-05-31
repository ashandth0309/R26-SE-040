"""
Face Recognition Module - Working with OpenCV face detection
"""

import cv2
import numpy as np
import time

class FaceRecognition:
    def __init__(self):
        print("🔄 Initializing Face Recognition Module...")
        
        # Use OpenCV's face detector (fast and reliable)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("❌ Failed to load face cascade!")
        else:
            print("✅ Face cascade loaded successfully")
        
        self.min_face_size = (60, 60)  # Minimum face size to detect
        self.scale_factor = 1.1
        self.min_neighbors = 5
        
        # Cache for performance
        self.last_detection_time = 0
        self.detection_interval = 0.05  # 50ms between detections
        self.cached_faces = []
        
        print("✅ Face Recognition Module Ready")
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using OpenCV Haar Cascade
        Returns list of (x, y, w, h) bounding boxes
        """
        if frame is None or frame.size == 0:
            return []
        
        current_time = time.time()
        
        # Return cached results if within interval
        if current_time - self.last_detection_time < self.detection_interval:
            return self.cached_faces
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size
            )
            
            # Convert to list of tuples
            detected_faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
            
            # Cache results
            self.cached_faces = detected_faces
            self.last_detection_time = current_time
            
            return detected_faces
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def extract_face(self, frame, bbox):
        """Extract face ROI from frame"""
        try:
            x, y, w, h = bbox
            
            # Ensure bounds are within frame
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return None
            
            # Extract face
            face = frame[y:y+h, x:x+w]
            
            if face is None or face.size == 0:
                return None
            
            # Resize to standard size
            face = cv2.resize(face, (100, 100))
            
            return face
            
        except Exception as e:
            return None
    
    def get_embedding(self, face_img):
        """
        Generate simple face embedding using HOG-like features
        This is fast and works for basic face recognition
        """
        if face_img is None:
            return None
        
        try:
            # Convert to grayscale
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            # Resize to fixed size
            gray = cv2.resize(gray, (64, 64))
            
            embedding = []
            
            # 1. Global features
            embedding.append(np.mean(gray))
            embedding.append(np.std(gray))
            
            # 2. Divide into 4x4 grid and get mean/std for each cell
            h, w = gray.shape
            cell_h, cell_w = h // 4, w // 4
            
            for i in range(4):
                for j in range(4):
                    y1, y2 = i * cell_h, (i + 1) * cell_h
                    x1, x2 = j * cell_w, (j + 1) * cell_w
                    cell = gray[y1:y2, x1:x2]
                    if cell.size > 0:
                        embedding.append(np.mean(cell))
                        embedding.append(np.std(cell))
                    else:
                        embedding.append(0)
                        embedding.append(0)
            
            # 3. Histogram features
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist.flatten() / np.sum(hist)
            embedding.extend(hist.tolist())
            
            # Convert to numpy array
            embedding = np.array(embedding, dtype=np.float32)
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def get_embeddings_batch(self, faces):
        """Generate embeddings for multiple faces"""
        if not faces:
            return []
        
        embeddings = []
        for face in faces:
            emb = self.get_embedding(face)
            if emb is not None:
                embeddings.append(emb)
        
        return embeddings
    
    def compare_faces(self, embedding1, embedding2):
        """Compare two face embeddings using cosine similarity"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
        else:
            similarity = 0
        
        return similarity
    
    def validate_face(self, face):
        """Validate if face image is good quality"""
        if face is None or face.size == 0:
            return False
        
        h, w = face.shape[:2]
        if h < 50 or w < 50:
            return False
        
        return True