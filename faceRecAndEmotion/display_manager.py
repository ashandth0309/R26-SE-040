"""
Display Manager for Robot Screen Interface
Handles all visual output and GUI elements
Optimized version for maximum performance and frame smoothness
"""

import cv2
import numpy as np
from datetime import datetime  # Moved to top level to avoid per-frame import lag
from utils.helpers import draw_text_with_background
from config import (DISPLAY_SCALE, FONT_SCALE, FONT_THICKNESS, 
                    TEXT_COLOR, BOX_COLOR, UNKNOWN_BOX_COLOR)

class DisplayManager:
    def __init__(self):
        self.scale = DISPLAY_SCALE
        self.font_scale = FONT_SCALE
        self.thickness = FONT_THICKNESS
        self.text_color = TEXT_COLOR
        self.known_color = BOX_COLOR
        self.unknown_color = UNKNOWN_BOX_COLOR
        
        # Display info cache
        self.current_name = None
        self.current_emotion = None
        self.current_confidence = None
        self.current_face_box = None
        self.is_known = False
        
        # UI Elements
        self.show_emotion_bars = False
        self.show_fps = True
        self.show_status = True
        
        # Colors for different emotions
        self.emotion_colors = {
            "happy": (0, 255, 0),       # Green
            "sad": (255, 0, 0),         # Blue
            "angry": (0, 0, 255),       # Red
            "fear": (128, 0, 128),      # Purple
            "surprise": (255, 255, 0),  # Cyan
            "disgust": (0, 255, 255),   # Yellow
            "neutral": (128, 128, 128)  # Gray
        }
        
        # ASCII fallback identifiers for standard cv2.putText compatibility
        self.emotion_labels = {
            "happy": "[HAPPY]",
            "sad": "[SAD]",
            "angry": "[ANGRY]",
            "fear": "[FEAR]",
            "surprise": "[SURPRISE]",
            "disgust": "[DISGUST]",
            "neutral": "[NEUTRAL]"
        }
        
        print("✅ Display Manager initialized with loop optimizations")
    
    def draw_face_info(self, frame, face_bbox, name, emotion, confidence, is_known=True):
        """
        Draw face bounding box and information on frame
        """
        x, y, w, h = face_bbox
        
        # Choose color based on known/unknown
        box_color = self.known_color if is_known else self.unknown_color
        emotion_color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw bounding box with optimized rounded corners
        self._draw_rounded_rectangle(frame, (x, y), (x + w, y + h), box_color, self.thickness)
        
        # Get standard text indicator
        tag = self.emotion_labels.get(emotion, "[?]")
        
        # Prepare info text safely without causing encoding processing drops
        if is_known:
            info_text = f"{tag} {name} - {emotion.upper()} ({confidence:.1f}%)"
        else:
            info_text = f"{tag} UNKNOWN - {emotion.upper()} ({confidence:.1f}%)"
        
        # Draw text with background
        draw_text_with_background(
            frame, 
            info_text, 
            (x, y - 10),
            font_scale=self.font_scale,
            thickness=self.thickness,
            text_color=self.text_color,
            bg_color=box_color,
            alpha=0.7
        )
        
        # Add confidence bar
        self._draw_confidence_bar(frame, (x, y + h + 5), w, confidence, emotion_color)
        
        # Update cache
        self.current_face_box = (x, y, w, h)
        self.current_name = name
        self.current_emotion = emotion
        self.current_confidence = confidence
        self.is_known = is_known
        
        return frame
    
    def _draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, r=8):
        """Optimized corner-only bounding box for significantly faster processing frames"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw a clean box outline (much lighter math footprint than 2 Rectangles + 4 Circles per frame)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Add accented brackets for a sleek robotic UI look
        cv2.line(img, (x1, y1), (x1 + r, y1), color, thickness + 1)
        cv2.line(img, (x1, y1), (x1, y1 + r), color, thickness + 1)
        
        cv2.line(img, (x2, y1), (x2 - r, y1), color, thickness + 1)
        cv2.line(img, (x2, y1), (x2, y1 + r), color, thickness + 1)
        
        cv2.line(img, (x1, y2), (x1 + r, y2), color, thickness + 1)
        cv2.line(img, (x1, y2), (x1, y2 - r), color, thickness + 1)
        
        cv2.line(img, (x2, y2), (x2 - r, y2), color, thickness + 1)
        cv2.line(img, (x2, y2), (x2, y2 - r), color, thickness + 1)
    
    def _draw_confidence_bar(self, frame, position, width, confidence, color):
        """Draw confidence bar below face box"""
        x, y = position
        bar_height = 4
        bar_width = int(width * (confidence / 100.0))
        
        # Background bar
        cv2.rectangle(frame, (x, y), (x + width, y + bar_height), (60, 60, 60), -1)
        
        # Confidence bar
        if bar_width > 0:
            cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), color, -1)
    
    def draw_status_bar(self, frame, status_text="System Running", fps=30):
        """
        Draw status bar at the bottom of frame with optimized processing
        """
        h, w = frame.shape[:2]
        bar_height = 35
        
        # Optimizing alpha composition calculation to prevent memory copy leaks
        roi = frame[h - bar_height:h, 0:w]
        overlay = np.zeros_like(roi)
        cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, roi)
        
        # Add status text
        cv2.putText(frame, f"SYS: {status_text}", (10, h - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add FPS with color coding
        fps_color = (0, 255, 0) if fps > 22 else (0, 255, 255) if fps > 12 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, h - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA)
        
        # Optimized Timestamp retrieval
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w // 2 - 35, h - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame
    
    def draw_enrollment_mode(self, frame, name, progress):
        """
        Draw enrollment mode interface
        """
        h, w = frame.shape[:2]
        
        # Draw lightweight backdrop overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        
        # Draw enrollment text
        cv2.putText(frame, "ENROLLMENT MODE", (w // 2 - 120, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Target: {name}", (w // 2 - 80, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw progress bar
        bar_width = 400
        bar_height = 30
        bar_x = (w - bar_width) // 2
        bar_y = h // 2
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (70, 70, 70), -1)
        
        # Progress fill
        fill_width = int(bar_width * (progress / 100.0))
        if fill_width > 0:
            color = (0, 0, 255) if progress < 33 else (0, 255, 255) if progress < 66 else (0, 255, 0)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Progress text
        cv2.putText(frame, f"{progress:.1f}%", (bar_x + bar_width // 2 - 20, bar_y + 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Instructions
        cv2.putText(frame, "Hold still and look directly into the camera lens", (w // 2 - 160, bar_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        
        frames_captured = int(progress * 0.5)
        cv2.putText(frame, f"Captured: {frames_captured}/50", (w // 2 - 60, bar_y + 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        return frame
    
    def create_emotion_display(self, emotion, probabilities):
        """
        Create a separate window side-panel with emotion probabilities
        """
        if probabilities is None:
            return None
        
        # Create a pre-allocated blank image for emotion panel
        display = np.zeros((320, 400, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(display, "Emotion Analysis Engine", (90, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        y_offset = 65
        for emotion_name, prob in probabilities.items():
            bar_width = int(prob * 2.2)  # Balanced width allocation
            color = self.emotion_colors.get(emotion_name, (128, 128, 128))
            
            # Bar background
            cv2.rectangle(display, (130, y_offset - 12), (130 + 220, y_offset + 8), (40, 40, 40), -1)
            
            # Actual metrics fill
            if bar_width > 0:
                cv2.rectangle(display, (130, y_offset - 12), (130 + bar_width, y_offset + 8), color, -1)
            
            # Text rendering
            lbl = self.emotion_labels.get(emotion_name, "")
            cv2.putText(display, f"{lbl} {prob:.1f}%", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
            y_offset += 32
        
        # Add active dominant label
        cv2.putText(display, f"Dominant State: {emotion.upper()}", (110, 295), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        return display
    
    def draw_system_info(self, frame, robot_status=None):
        """
        Draw system information overlay
        """
        y_offset = 25
        infos = [
            f"Robot State: {robot_status.get('emotion', 'N/A').upper() if robot_status else 'ACTIVE'}",
            f"User: {self.current_name if self.current_name else 'Searching...'}",
            f"Emotion: {self.current_emotion.upper() if self.current_emotion else 'N/A'}"
        ]
        
        for info in infos:
            draw_text_with_background(
                frame, info, (10, y_offset),
                font_scale=0.45,
                thickness=1,
                bg_color=(0, 0, 0),
                alpha=0.4
            )
            y_offset += 22
        
        return frame
    
    def get_emotion_color(self, emotion):
        return self.emotion_colors.get(emotion, (255, 255, 255))
    
    def toggle_emotion_bars(self):
        self.show_emotion_bars = not self.show_emotion_bars
        return self.show_emotion_bars
    
    def clear_cache(self):
        self.current_name = None
        self.current_emotion = None
        self.current_confidence = None
        self.current_face_box = None
        self.is_known = False