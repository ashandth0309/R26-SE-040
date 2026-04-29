"""
Display Manager for Robot Screen Interface
Handles all visual output and GUI elements
"""

import cv2
import numpy as np
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
            "happy": (0, 255, 0),      # Green
            "sad": (255, 0, 0),         # Blue
            "angry": (0, 0, 255),       # Red
            "fear": (128, 0, 128),      # Purple
            "surprise": (255, 255, 0),  # Cyan
            "disgust": (0, 255, 255),   # Yellow
            "neutral": (128, 128, 128)  # Gray
        }
        
        # Emoji mapping
        self.emotion_emojis = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "fear": "😨",
            "surprise": "😲",
            "disgust": "🤢",
            "neutral": "😐"
        }
        
        print("✅ Display Manager initialized")
    
    def draw_face_info(self, frame, face_bbox, name, emotion, confidence, is_known=True):
        """
        Draw face bounding box and information on frame
        """
        x, y, w, h = face_bbox
        
        # Choose color based on known/unknown
        box_color = self.known_color if is_known else self.unknown_color
        emotion_color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw bounding box with rounded corners
        self._draw_rounded_rectangle(frame, (x, y), (x + w, y + h), box_color, self.thickness)
        
        # Get emotion emoji
        emoji = self.emotion_emojis.get(emotion, "❓")
        
        # Prepare info text
        if is_known:
            info_text = f"{emoji} {name} - {emotion.upper()} ({confidence:.1f}%)"
        else:
            info_text = f"{emoji} UNKNOWN - {emotion.upper()} ({confidence:.1f}%)"
        
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
    
    def _draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, r=10):
        """Draw rectangle with rounded corners"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Rectangle corners
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        
        # Draw circles at corners
        cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
        cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
        cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
        cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)
    
    def _draw_confidence_bar(self, frame, position, width, confidence, color):
        """Draw confidence bar below face box"""
        x, y = position
        bar_height = 5
        bar_width = int(width * confidence / 100)
        
        # Background bar
        cv2.rectangle(frame, (x, y), (x + width, y + bar_height), (100, 100, 100), -1)
        
        # Confidence bar
        if bar_width > 0:
            cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), color, -1)
    
    def draw_status_bar(self, frame, status_text="System Running", fps=30):
        """
        Draw status bar at the bottom of frame
        """
        h, w = frame.shape[:2]
        bar_height = 40
        
        # Draw semi-transparent bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add status text
        cv2.putText(frame, f"⚡ {status_text}", (10, h - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add FPS with color coding
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(frame, f"📊 FPS: {fps:.1f}", (w - 120, h - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"🕐 {timestamp}", (w//2 - 50, h - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_enrollment_mode(self, frame, name, progress):
        """
        Draw enrollment mode interface
        progress: percentage of capture complete (0-100)
        """
        h, w = frame.shape[:2]
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw enrollment text
        cv2.putText(frame, f"📝 ENROLLMENT MODE", (w//2 - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Person: {name}", (w//2 - 100, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw progress bar
        bar_width = 500
        bar_height = 40
        bar_x = (w - bar_width) // 2
        bar_y = h // 2
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Progress fill
        fill_width = int(bar_width * progress / 100)
        if fill_width > 0:
            # Gradient color based on progress
            if progress < 33:
                color = (0, 0, 255)  # Red
            elif progress < 66:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         color, -1)
        
        # Progress text
        cv2.putText(frame, f"{progress:.1f}%", (bar_x + bar_width//2 - 30, bar_y + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Look at the camera and hold still", (w//2 - 150, bar_y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        frames_captured = int(progress * 50 / 100)
        cv2.putText(frame, f"Captured: {frames_captured}/50 frames", (w//2 - 100, bar_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def create_emotion_display(self, emotion, probabilities):
        """
        Create a separate window with emotion probabilities
        """
        if probabilities is None:
            return None
        
        # Create a blank image for emotion display
        display = np.zeros((350, 450, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(display, "Emotion Analysis", (130, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_offset = 70
        for emotion_name, prob in probabilities.items():
            # Draw bar
            bar_width = int(prob * 2.5)  # Scale to max 250px for 100%
            color = self.emotion_colors.get(emotion_name, (128, 128, 128))
            
            # Bar background
            cv2.rectangle(display, (120, y_offset - 15), 
                         (120 + 250, y_offset + 10), (50, 50, 50), -1)
            
            # Actual bar
            if bar_width > 0:
                cv2.rectangle(display, (120, y_offset - 15), 
                             (120 + bar_width, y_offset + 10), color, -1)
            
            # Draw text
            emoji = self.emotion_emojis.get(emotion_name, "")
            cv2.putText(display, f"{emoji} {emotion_name}: {prob:.1f}%", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            y_offset += 35
        
        # Add legend
        cv2.putText(display, f"Dominant: {emotion.upper()}", (150, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return display
    
    def draw_system_info(self, frame, robot_status=None):
        """
        Draw system information overlay
        """
        h, w = frame.shape[:2]
        
        # Top-left corner - System status
        y_offset = 30
        infos = [
            f"🤖 Robot Status: {robot_status.get('emotion', 'N/A') if robot_status else 'Active'}",
            f"👤 Current: {self.current_name if self.current_name else 'None'}",
            f"😊 Emotion: {self.current_emotion if self.current_emotion else 'None'}"
        ]
        
        for info in infos:
            draw_text_with_background(
                frame, info, (10, y_offset),
                font_scale=0.5,
                thickness=1,
                bg_color=(0, 0, 0),
                alpha=0.5
            )
            y_offset += 25
        
        return frame
    
    def get_emotion_color(self, emotion):
        """Get color for emotion bar"""
        return self.emotion_colors.get(emotion, (255, 255, 255))
    
    def toggle_emotion_bars(self):
        """Toggle emotion bars display"""
        self.show_emotion_bars = not self.show_emotion_bars
        return self.show_emotion_bars
    
    def clear_cache(self):
        """Clear display cache"""
        self.current_name = None
        self.current_emotion = None
        self.current_confidence = None
        self.current_face_box = None
        self.is_known = False