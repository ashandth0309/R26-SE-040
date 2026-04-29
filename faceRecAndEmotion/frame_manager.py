"""
Frame Manager for Circular Buffer Storage
Handles capture and management of unknown person frames
"""

import cv2
import os
import shutil
from datetime import datetime
from utils.helpers import ensure_dir, save_frame
from config import MAX_UNKNOWN_FRAMES, UNKNOWN_FRAMES_DIR, CAPTURE_EVERY_N_FRAMES

class FrameManager:
    def __init__(self):
        self.max_frames = MAX_UNKNOWN_FRAMES
        self.base_dir = UNKNOWN_FRAMES_DIR
        self.current_session_dir = None
        self.frames_buffer = []  # Stores paths to frames
        self.capture_count = 0
        self.is_capturing = False
        self.current_unknown_id = None
        
    def start_capture(self, unknown_id=None):
        """
        Start capturing frames for unknown person
        """
        self.is_capturing = True
        self.capture_count = 0
        self.frames_buffer = []
        
        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if unknown_id:
            self.current_unknown_id = unknown_id
            session_name = f"unknown_{unknown_id}_{timestamp}"
        else:
            session_name = f"unknown_{timestamp}"
        
        self.current_session_dir = os.path.join(self.base_dir, session_name)
        ensure_dir(self.current_session_dir)
        
        print(f"🎥 Started capturing frames for unknown person in {self.current_session_dir}")
        return self.current_session_dir
    
    def stop_capture(self):
        """
        Stop capturing frames
        """
        self.is_capturing = False
        session_dir = self.current_session_dir
        self.current_session_dir = None
        self.current_unknown_id = None
        print(f"🛑 Stopped capturing. Captured {len(self.frames_buffer)} frames")
        return session_dir
    
    def add_frame(self, frame, face_bbox):
        """
        Add frame to circular buffer
        Returns: True if frame was added
        """
        if not self.is_capturing:
            return False
        
        # Capture every N frames to reduce storage
        self.capture_count += 1
        if self.capture_count % CAPTURE_EVERY_N_FRAMES != 0:
            return False
        
        # Extract face region
        x, y, w, h = face_bbox
        face_frame = frame[y:y+h, x:x+w]
        
        if face_frame.size == 0:
            return False
        
        # Save frame
        frame_path = save_frame(face_frame, self.current_session_dir, 
                                f"unknown_face_{len(self.frames_buffer):03d}")
        
        # Add to circular buffer
        if len(self.frames_buffer) >= self.max_frames:
            # Remove oldest frame
            oldest_frame = self.frames_buffer.pop(0)
            try:
                os.remove(oldest_frame)
            except:
                pass
        
        self.frames_buffer.append(frame_path)
        
        print(f"📸 Captured frame {len(self.frames_buffer)}/{self.max_frames}")
        return True
    
    def get_captured_count(self):
        """
        Get number of captured frames
        """
        return len(self.frames_buffer)
    
    def is_buffer_full(self):
        """
        Check if buffer has reached maximum
        """
        return len(self.frames_buffer) >= self.max_frames
    
    def cleanup_old_sessions(self, keep_last_n=5):
        """
        Delete old unknown face sessions, keep only last N
        """
        sessions = [d for d in os.listdir(self.base_dir) 
                   if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith("unknown_")]
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: os.path.getctime(os.path.join(self.base_dir, x)), reverse=True)
        
        # Delete old sessions
        for session in sessions[keep_last_n:]:
            session_path = os.path.join(self.base_dir, session)
            shutil.rmtree(session_path)
            print(f"🧹 Deleted old session: {session}")