"""
Frame Manager for Circular Buffer Storage
Handles capture and management of unknown person frames
Optimized version with asynchronous background I/O threads to eliminate loop lag
"""

import cv2
import os
import shutil
import threading  # Added to prevent synchronous disk I/O blocking our main camera loop
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
        Add frame to circular buffer asynchronously to guarantee fluid video rendering speeds
        """
        if not self.is_capturing or frame is None:
            return False
        
        # Capture every N frames to scale performance gracefully
        self.capture_count += 1
        if self.capture_count % CAPTURE_EVERY_N_FRAMES != 0:
            return False
        
        try:
            # Safely clamp boundary coordinates before extracting array dimensions
            fh, fw = frame.shape[:2]
            x, y, w, h = face_bbox
            
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(fw, x + w)
            y2 = min(fh, y + h)
            
            face_frame = frame[y1:y2, x1:x2]
            
            if face_frame.size == 0:
                return False
            
            # Make a clean memory copy of the cropped array slice so the background thread can safely process it
            face_frame_copy = face_frame.copy()
            frame_index = len(self.frames_buffer)
            
            # Handle the circular buffer pruning mechanics locally
            oldest_frame_to_remove = None
            if len(self.frames_buffer) >= self.max_frames:
                oldest_frame_to_remove = self.frames_buffer.pop(0)
            
            # Pre-calculate what the path *will* be once written to update the index instantly
            predicted_filename = f"unknown_face_{frame_index:03d}"
            predicted_path = os.path.join(self.current_session_dir, f"{predicted_filename}.jpg")
            self.frames_buffer.append(predicted_path)

            # Delegate disk operations seamlessly to a background worker thread
            def _async_file_io(img, target_dir, file_name, file_to_delete):
                try:
                    save_frame(img, target_dir, file_name)
                    if file_to_delete and os.path.exists(file_to_delete):
                        os.remove(file_to_delete)
                except Exception as ex:
                    print(f"⚠️ Non-critical Frame I/O thread warning: {ex}")

            threading.Thread(
                target=_async_file_io, 
                args=(face_frame_copy, self.current_session_dir, predicted_filename, oldest_frame_to_remove), 
                daemon=True
            ).start()
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error preparing frame crop: {e}")
            return False
    
    def get_captured_count(self):
        return len(self.frames_buffer)
    
    def is_buffer_full(self):
        return len(self.frames_buffer) >= self.max_frames
    
    def cleanup_old_sessions(self, keep_last_n=5):
        """
        Delete old unknown face sessions inside an isolated thread context to prevent startup lag
        """
        def _cleanup():
            try:
                if not os.path.exists(self.base_dir):
                    return
                sessions = [d for d in os.listdir(self.base_dir) 
                           if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith("unknown_")]
                
                # Sort by creation time (newest first)
                sessions.sort(key=lambda x: os.path.getctime(os.path.join(self.base_dir, x)), reverse=True)
                
                # Delete old sessions
                for session in sessions[keep_last_n:]:
                    session_path = os.path.join(self.base_dir, session)
                    shutil.rmtree(session_path)
                    print(f"🧹 Deleted old session: {session}")
            except Exception as e:
                print(f"⚠️ Cleanup thread encountered a minor issue: {e}")

        threading.Thread(target=_cleanup, daemon=True).start()