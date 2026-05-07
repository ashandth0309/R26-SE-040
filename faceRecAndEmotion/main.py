"""
Main System - Smart AI Dog Robot
Face Recognition + Emotion Detection System
COMPLETE FIXED VERSION with Windows camera support
"""

import cv2
import numpy as np
import time
import argparse
import os
from datetime import datetime

# Import all modules
from config import *
from face_recognition_module import FaceRecognition
from emotion_detection import EmotionDetector
from database_manager import DatabaseManager
from frame_manager import FrameManager
from display_manager import DisplayManager
from voice_manager import VoiceManager
from robot_controller import RobotController
from emotion_logger import EmotionLogger
from utils.helpers import get_timestamp

class SmartAIDogRobot:
    def __init__(self):
        print("="*50)
        print("🐶 SMART AI DOG ROBOT - INITIALIZING")
        print("="*50)
        
        # Initialize all modules
        try:
            self.face_recognizer = FaceRecognition()
            self.emotion_detector = EmotionDetector()
            self.database = DatabaseManager()
            self.frame_manager = FrameManager()
            self.display = DisplayManager()
            self.voice = VoiceManager()
            self.robot = RobotController()
            self.emotion_logger = EmotionLogger()
        except Exception as e:
            print(f"❌ Error initializing modules: {e}")
            raise
        
        # Camera setup
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # System state
        self.is_running = False
        self.enrollment_mode = False
        self.enrollment_name = None
        self.enrollment_frames = []
        self.enrollment_bbox = None
        
        # Recognition cache
        self.recognized_faces = {}  # Track recently recognized faces
        self.recognition_cooldown = 2  # seconds
        
        # Frame processing throttle
        self.last_process_time = 0
        self.process_interval = 0.1  # Process every 100ms
        
        print("✅ System initialized successfully!")
        print("="*50)
    
    def start_camera(self):
        """Start camera capture with Windows compatibility"""
        print("📷 Attempting to start camera...")
        
        # Try different camera backends in order of preference
        backends = [
            (cv2.CAP_DSHOW, "DirectShow (Windows)"),
            (cv2.CAP_ANY, "Auto-detect"),
            (cv2.CAP_MSMF, "Microsoft Media Foundation"),
            (cv2.CAP_VFW, "Video for Windows")
        ]
        
        # Try different camera IDs
        camera_ids = [0]
        if isinstance(CAMERA_ID, str) and CAMERA_ID.startswith('http'):
            camera_ids = [CAMERA_ID]  # IP camera
        
        for cam_id in camera_ids:
            for backend, backend_name in backends:
                try:
                    print(f"  Trying camera {cam_id} with {backend_name}...")
                    
                    # Open camera with specific backend
                    if backend != cv2.CAP_ANY:
                        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    else:
                        self.cap = cv2.VideoCapture(cam_id)
                    
                    if not self.cap or not self.cap.isOpened():
                        continue
                    
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                    
                    # Give camera time to initialize
                    time.sleep(0.5)
                    
                    # Test frame capture
                    for attempt in range(3):
                        ret, frame = self.cap.read()
                        if ret and frame is not None and frame.size > 0:
                            print(f"  ✅ Success! Camera ID: {cam_id}, Backend: {backend_name}")
                            print(f"     Frame size: {frame.shape}")
                            
                            # Store successful settings for future use
                            self.camera_settings = {
                                'id': cam_id,
                                'backend': backend,
                                'backend_name': backend_name
                            }
                            return True
                        time.sleep(0.1)
                    
                    # If we get here, frame capture failed
                    self.cap.release()
                    
                except Exception as e:
                    print(f"  ❌ Error: {str(e)[:50]}")
                    if self.cap:
                        self.cap.release()
                    continue
        
        print("❌ Error: Could not open camera with any backend")
        print("\nTroubleshooting tips:")
        print("1. Check if camera is connected and not used by another app")
        print("2. Run test_camera.py to diagnose")
        print("3. Try using a USB camera (set CAMERA_ID = 1)")
        print("4. Check Windows camera permissions")
        return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            print("📷 Camera stopped")
    
    def process_frame(self, frame):
        """Process a single frame with throttling"""
        current_time = time.time()
        
        # Throttle processing to improve performance
        if current_time - self.last_process_time < self.process_interval:
            return frame
        
        try:
            # Detect faces
            faces = self.face_recognizer.detect_faces(frame)
            
            # Process each face
            for bbox in faces:
                x, y, w, h = bbox
                
                # Extract face region
                face_img = self.face_recognizer.extract_face(frame, bbox)
                
                if face_img is None or face_img.size == 0:
                    continue
                
                # Get face embedding
                embedding = self.face_recognizer.get_embedding(face_img)
                
                if embedding is None:
                    continue
                
                # Find person in database
                name, similarity = self.database.find_person(embedding)
                
                # Detect emotion
                emotion, emotion_probs = self.emotion_detector.detect_emotion(face_img)
                
                if name:
                    # Known person
                    is_known = True
                    
                    # Check recognition cooldown
                    if name not in self.recognized_faces or \
                       current_time - self.recognized_faces[name] > self.recognition_cooldown:
                        
                        # Log emotion
                        self.emotion_logger.log_emotion(name, emotion, 
                                                       emotion_probs.get(emotion, 0))
                        
                        # Robot reaction
                        self.robot.react_to_emotion(emotion, name)
                        
                        # Voice greeting (only occasionally)
                        if np.random.random() < 0.3:  # 30% chance
                            self.voice.greet_person(name, emotion)
                        
                        # Update recognition cache
                        self.recognized_faces[name] = current_time
                    
                    # Display info
                    frame = self.display.draw_face_info(
                        frame, bbox, name, emotion, 
                        emotion_probs.get(emotion, 0), is_known
                    )
                    
                else:
                    # Unknown person
                    is_known = False
                    
                    # Start capturing frames if not already
                    if not self.frame_manager.is_capturing:
                        self.frame_manager.start_capture()
                        self.voice.greet_unknown()
                    
                    # Add frame to buffer
                    self.frame_manager.add_frame(frame, bbox)
                    
                    # Check if buffer is full
                    if self.frame_manager.is_buffer_full():
                        print("✅ Captured 50 frames of unknown person")
                        self.frame_manager.stop_capture()
                    
                    # Display info
                    frame = self.display.draw_face_info(
                        frame, bbox, "UNKNOWN", emotion, 
                        emotion_probs.get(emotion, 0), is_known
                    )
            
            self.last_process_time = current_time
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return frame
    
    def run(self):
        """Main system loop"""
        if not self.start_camera():
            print("\n❌ Cannot start camera. Exiting...")
            return
        
        self.is_running = True
        
        print("\n🚀 System running!")
        print("Controls:")
        print("  'q' - Quit")
        print("  'e' - Enrollment mode (add new person)")
        print("  'c' - Cancel enrollment")
        print("  's' - Show statistics")
        print("-" * 50)
        
        # Warm-up frames
        print("Warming up camera...")
        for i in range(10):
            ret, frame = self.cap.read()
            if not ret:
                break
            time.sleep(0.05)
        print("Ready!\n")
        
        while self.is_running:
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("⚠️ Lost frame, attempting to reconnect...")
                    time.sleep(0.5)
                    continue
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Make a copy for display
                display_frame = frame.copy()
                
                if self.enrollment_mode:
                    # Enrollment mode
                    display_frame = self.display.draw_enrollment_mode(
                        display_frame, 
                        self.enrollment_name,
                        len(self.enrollment_frames) / 50 * 100
                    )
                    
                    # Detect face for enrollment
                    faces = self.face_recognizer.detect_faces(frame)
                    
                    if faces and len(self.enrollment_frames) < 50:
                        bbox = faces[0]  # Use first face
                        self.enrollment_bbox = bbox
                        
                        # Extract and save face
                        face_img = self.face_recognizer.extract_face(frame, bbox)
                        if face_img is not None and face_img.size > 0:
                            self.enrollment_frames.append(frame.copy())
                            print(f"📸 Enrollment frame {len(self.enrollment_frames)}/50", end='\r')
                        
                        # Draw bounding box
                        x, y, w, h = bbox
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), 
                                     (255, 255, 0), 2)
                    
                    # Check if enrollment complete
                    if len(self.enrollment_frames) >= 50:
                        print()  # New line after progress
                        self.complete_enrollment()
                else:
                    # Normal recognition mode
                    display_frame = self.process_frame(display_frame)
                
                # Draw status bar
                display_frame = self.display.draw_status_bar(
                    display_frame, 
                    "Enrollment Mode" if self.enrollment_mode else "Recognition Mode",
                    self.fps
                )
                
                # Show frame
                cv2.imshow("🐶 Smart AI Dog Robot", display_frame)
                
                # Handle keyboard input (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('e'):
                    self.start_enrollment()
                elif key == ord('c'):
                    self.cancel_enrollment()
                elif key == ord('s'):
                    self.show_stats()
                    
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
        
        # Cleanup
        self.cleanup()
    
    def start_enrollment(self):
        """Start enrollment mode for new person"""
        if self.enrollment_mode:
            print("Already in enrollment mode")
            return
        
        print("\n" + "="*50)
        print("📝 ENROLLMENT MODE")
        print("="*50)
        name = input("Enter name for new person: ").strip()
        
        if name and len(name) > 0:
            self.enrollment_mode = True
            self.enrollment_name = name
            self.enrollment_frames = []
            self.enrollment_bbox = None
            print(f"✅ Enrollment started for: {name}")
            print("🎥 Look at the camera and hold still...")
            print("📸 Capturing 50 frames...")
        else:
            print("❌ Invalid name")
    
    def complete_enrollment(self):
        """Complete enrollment and save to database"""
        print("\n" + "="*50)
        print("📊 PROCESSING ENROLLMENT")
        print("="*50)
        
        if len(self.enrollment_frames) < 10:
            print("❌ Not enough frames captured (minimum 10)")
            self.cancel_enrollment()
            return
        
        print(f"Processing {len(self.enrollment_frames)} frames...")
        
        # Extract embeddings from all frames
        embeddings = self.face_recognizer.extract_embeddings_from_frames(
            self.enrollment_frames, 
            self.enrollment_bbox
        )
        
        if len(embeddings) > 5:  # At least 5 successful embeddings
            # Add to database
            person_id = self.database.add_person(self.enrollment_name, embeddings)
            print(f"✅ Person '{self.enrollment_name}' enrolled successfully!")
            print(f"   ID: {person_id}")
            print(f"   Embeddings: {len(embeddings)}")
            
            # Voice feedback
            try:
                self.voice.speak(f"Welcome {self.enrollment_name}! I've learned your face.")
            except:
                pass
        else:
            print(f"❌ Failed to extract enough face embeddings (got {len(embeddings)})")
            print("   Please try again with better lighting")
        
        # Exit enrollment mode
        self.enrollment_mode = False
        self.enrollment_name = None
        self.enrollment_frames = []
        print("="*50 + "\n")
    
    def cancel_enrollment(self):
        """Cancel enrollment mode"""
        if self.enrollment_mode:
            self.enrollment_mode = False
            self.enrollment_name = None
            self.enrollment_frames = []
            print("❌ Enrollment cancelled")
    
    def show_stats(self):
        """Show system statistics"""
        print("\n" + "="*50)
        print("📊 SYSTEM STATISTICS")
        print("="*50)
        
        # Known persons
        persons = self.database.get_all_persons()
        print(f"\n👤 Known Persons: {len(persons)}")
        for p in persons:
            print(f"  - {p['name']} (ID: {p['id']})")
        
        # Emotion stats
        try:
            stats = self.emotion_logger.get_daily_summary()
            if stats:
                print(f"\n📈 Today's Summary:")
                print(f"  Total detections: {stats['total_detections']}")
                print(f"  Most common: {stats['most_common_emotion']}")
                print(f"  Average confidence: {stats['average_confidence']:.1f}%")
        except Exception as e:
            print(f"\n⚠️ Could not load emotion stats: {e}")
        
        # Unknown captures
        try:
            unknown_sessions = [d for d in os.listdir(UNKNOWN_FRAMES_DIR) 
                               if d.startswith("unknown_")]
            print(f"\n📸 Unknown Person Sessions: {len(unknown_sessions)}")
        except:
            print(f"\n📸 Unknown Person Sessions: 0")
        
        # Camera info
        if hasattr(self, 'camera_settings'):
            print(f"\n📷 Camera: ID {self.camera_settings['id']} ({self.camera_settings['backend_name']})")
        
        # Performance
        print(f"\n⚡ Performance:")
        print(f"  FPS: {self.fps:.1f}")
        print(f"  Uptime: {int(time.time() - self.start_time)} seconds")
        
        print("="*50 + "\n")
    
    def cleanup(self):
        """Cleanup system resources"""
        print("\n🧹 Cleaning up...")
        
        self.is_running = False
        
        # Stop camera
        self.stop_camera()
        
        # Stop frame capture if active
        if self.frame_manager and self.frame_manager.is_capturing:
            self.frame_manager.stop_capture()
        
        # Cleanup robot
        if hasattr(self, 'robot'):
            self.robot.cleanup()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        print("✅ System shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Smart AI Dog Robot")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--camera", type=int, default=None, help="Camera ID to use")
    args = parser.parse_args()
    
    # Override camera ID if specified
    if args.camera is not None:
        global CAMERA_ID
        CAMERA_ID = args.camera
    
    print("🐶 Smart AI Dog Robot - Starting...")
    print("="*50)
    
    # Create and run robot
    robot = None
    try:
        robot = SmartAIDogRobot()
        
        if args.test:
            print("\n🔧 Running in TEST MODE")
            print("Testing camera only...")
            if robot.start_camera():
                print("✅ Camera test passed")
                time.sleep(2)
            else:
                print("❌ Camera test failed")
        else:
            robot.run()
            
    except KeyboardInterrupt:
        print("\n⚠️ System interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot:
            robot.cleanup()
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()