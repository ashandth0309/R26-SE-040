"""
Debug Face Recognition - Find why it's not working
FIXED for your DatabaseManager structure
"""

import cv2
import numpy as np
import json
from pathlib import Path

# Import your modules
from face_recognition_module import FaceRecognition
from database_manager import DatabaseManager
from config import SIMILARITY_THRESHOLD

def debug_recognition():
    print("=" * 60)
    print("🔍 FACE RECOGNITION DEBUG TOOL")
    print("=" * 60)
    
    # Initialize
    recognizer = FaceRecognition()
    db = DatabaseManager()
    
    # Check database
    print("\n📂 DATABASE CHECK:")
    print(f"   Total persons in DB: {len(db.known_faces['persons'])}")
    
    if len(db.known_faces['persons']) == 0:
        print("\n❌ NO PERSONS IN DATABASE!")
        print("   Please enroll a person first using 'e' key in main program")
        return
    
    for i, person in enumerate(db.known_faces['persons']):
        print(f"\n   👤 Person {i+1}: {person['name']}")
        print(f"      - ID: {person['id']}")
        print(f"      - Embeddings count: {len(person.get('embeddings', []))}")
        print(f"      - Created at: {person.get('created_at', 'Unknown')}")
        
        # Check avg_embedding
        if 'avg_embedding' in person:
            avg_emb = person['avg_embedding']
            print(f"      - Avg embedding shape: {len(avg_emb)}")
            print(f"      - Avg embedding sample: {avg_emb[:5]}...")
    
    # Open camera
    print("\n📷 Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    print("\n✅ Camera opened. Looking for faces...")
    print("   Press 'c' to capture and test current face")
    print("   Press 't' to test with sample from database")
    print("   Press 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Detect faces
        faces = recognizer.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Draw box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face
            face_img = recognizer.extract_face(frame, (x, y, w, h))
            
            if face_img is not None:
                # Validate face
                is_valid = recognizer.validate_face(face_img)
                
                if is_valid:
                    # Get embedding
                    embedding = recognizer.get_embedding(face_img)
                    
                    if embedding is not None:
                        # Try to find person
                        name, similarity = db.find_person(embedding)
                        
                        # Display result
                        if name:
                            label = f"✅ {name} ({similarity:.3f})"
                            color = (0, 255, 0)
                        else:
                            label = f"❌ UNKNOWN ({similarity:.3f})"
                            color = (0, 0, 255)
                        
                        cv2.putText(frame, label, (x, y+h+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        cv2.putText(frame, "❌ No embedding", (x, y+h+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(frame, "⚠️ Invalid face", (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show info
        cv2.putText(frame, f"Threshold: {SIMILARITY_THRESHOLD}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Debug: 'c'=test, 't'=test sample, 'q'=quit", (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Face Recognition Debug", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("\n📸 Capture requested - testing current face...")
            if len(faces) > 0:
                test_face_deep(recognizer, db, frame, faces[0])
            else:
                print("   ❌ No face detected! Please look at camera.")
        elif key == ord('t'):
            print("\n🔬 Testing with database samples...")
            test_database_samples(recognizer, db)
    
    cap.release()
    cv2.destroyAllWindows()

def test_face_deep(recognizer, db, frame, bbox):
    """Deep test of a single face"""
    print("\n" + "=" * 60)
    print("🔬 DEEP FACE ANALYSIS")
    print("=" * 60)
    
    x, y, w, h = bbox
    face_img = recognizer.extract_face(frame, (x, y, w, h))
    
    if face_img is None:
        print("❌ Could not extract face")
        return
    
    # Show face in window
    print("📸 Displaying extracted face... Press any key to continue")
    cv2.imshow("Extracted Face - Press any key", face_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Extracted Face - Press any key")
    
    # Validate face
    is_valid = recognizer.validate_face(face_img)
    print(f"\n✅ Face validation: {'PASS' if is_valid else 'FAIL'}")
    
    if not is_valid:
        h, w = face_img.shape[:2]
        print(f"   Size: {w}x{h}")
        return
    
    # Get embedding
    embedding = recognizer.get_embedding(face_img)
    
    if embedding is None:
        print("❌ Could not generate embedding")
        return
    
    print(f"\n📊 Embedding Info:")
    print(f"   Shape: {embedding.shape}")
    print(f"   Type: {embedding.dtype}")
    print(f"   Mean: {np.mean(embedding):.6f}")
    print(f"   Std: {np.std(embedding):.6f}")
    print(f"   Norm: {np.linalg.norm(embedding):.6f}")
    print(f"   Min: {np.min(embedding):.6f}")
    print(f"   Max: {np.max(embedding):.6f}")
    
    # Compare with all persons in database
    print(f"\n🎯 COMPARING WITH DATABASE (Threshold: {SIMILARITY_THRESHOLD}):")
    print("-" * 50)
    
    best_match = None
    best_score = -1
    
    for person in db.known_faces["persons"]:
        if "avg_embedding" in person:
            avg_emb = np.array(person["avg_embedding"])
            
            # Calculate cosine similarity
            dot = np.dot(embedding, avg_emb)
            norm1 = np.linalg.norm(embedding)
            norm2 = np.linalg.norm(avg_emb)
            
            if norm1 > 0 and norm2 > 0:
                similarity = dot / (norm1 * norm2)
            else:
                similarity = 0
            
            # Also calculate Euclidean distance
            euclidean = np.linalg.norm(embedding - avg_emb)
            
            print(f"\n   👤 {person['name']}:")
            print(f"      Cosine Similarity: {similarity:.4f}")
            print(f"      Euclidean Distance: {euclidean:.4f}")
            print(f"      Match: {'✅ YES' if similarity >= SIMILARITY_THRESHOLD else '❌ NO'}")
            
            if similarity > best_score:
                best_score = similarity
                best_match = person['name']
    
    print("\n" + "=" * 50)
    print(f"🏆 BEST MATCH: {best_match}")
    print(f"📊 BEST SCORE: {best_score:.4f}")
    print(f"🎯 THRESHOLD: {SIMILARITY_THRESHOLD}")
    
    if best_score >= SIMILARITY_THRESHOLD:
        print(f"\n✅✅✅ RECOGNIZED: {best_match} ✅✅✅")
    else:
        print(f"\n❌❌❌ UNKNOWN - Score {best_score:.4f} below threshold {SIMILARITY_THRESHOLD} ❌❌❌")
        
        # Calculate recommended threshold
        recommended = max(0.25, best_score - 0.05)
        print(f"\n💡 RECOMMENDED FIXES:")
        print(f"   1. Change threshold in config.py to: {recommended:.2f}")
        print(f"   2. Or re-enroll with better lighting and more samples")
        print(f"   3. Make sure you're looking directly at camera")
        print(f"   4. Remove glasses/hat if possible")

def test_database_samples(recognizer, db):
    """Test using samples stored in database"""
    print("\n" + "=" * 60)
    print("🔬 TESTING DATABASE SAMPLES")
    print("=" * 60)
    
    for person in db.known_faces["persons"]:
        print(f"\n👤 Testing: {person['name']}")
        
        if 'embeddings' not in person or len(person['embeddings']) == 0:
            print("   ❌ No embeddings in database")
            continue
        
        # Test first embedding against avg_embedding
        first_emb = np.array(person['embeddings'][0])
        avg_emb = np.array(person['avg_embedding'])
        
        # Calculate similarity
        dot = np.dot(first_emb, avg_emb)
        norm1 = np.linalg.norm(first_emb)
        norm2 = np.linalg.norm(avg_emb)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot / (norm1 * norm2)
        else:
            similarity = 0
        
        print(f"   Self-similarity (first vs avg): {similarity:.4f}")
        
        if similarity < 0.5:
            print(f"   ⚠️ WARNING: Low self-similarity! Embeddings are inconsistent.")
            print(f"   This means enrollment quality is poor.")
        
        # Test against other persons
        for other in db.known_faces["persons"]:
            if other['name'] == person['name']:
                continue
            
            other_avg = np.array(other['avg_embedding'])
            dot = np.dot(first_emb, other_avg)
            norm1 = np.linalg.norm(first_emb)
            norm2 = np.linalg.norm(other_avg)
            
            if norm1 > 0 and norm2 > 0:
                cross_sim = dot / (norm1 * norm2)
            else:
                cross_sim = 0
            
            print(f"   Cross-similarity with {other['name']}: {cross_sim:.4f}")

if __name__ == "__main__":
    debug_recognition()