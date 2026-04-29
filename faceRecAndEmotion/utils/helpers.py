"""
Helper utilities for the AI Dog Robot
"""

import json
import numpy as np
import cv2
from datetime import datetime
import os
import pickle

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def draw_text_with_background(img, text, position, font_scale=0.7, 
                             thickness=2, text_color=(255,255,255), 
                             bg_color=(0,0,0), alpha=0.6):
    """
    Draw text with semi-transparent background
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Make sure coordinates are within image bounds
    x = max(0, x)
    y = max(text_height + 10, y)
    
    # Calculate background rectangle coordinates
    bg_x1 = x - 5
    bg_y1 = y - text_height - 10
    bg_x2 = x + text_width + 10
    bg_y2 = y + baseline + 5
    
    # Ensure rectangle is within image bounds
    bg_x1 = max(0, bg_x1)
    bg_y1 = max(0, bg_y1)
    bg_x2 = min(img.shape[1], bg_x2)
    bg_y2 = min(img.shape[0], bg_y2)
    
    # Draw background rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw text
    cv2.putText(img, text, (x, y - 5), font, font_scale, text_color, thickness)
    
    return img

def ensure_dir(directory):
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)

def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def save_frame(frame, directory, prefix="frame"):
    """Save frame with timestamp"""
    ensure_dir(directory)
    timestamp = get_timestamp()
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, frame)
    return filepath

def load_json(filepath, default=None):
    """Load JSON file with error handling"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return default if default is not None else {}

def save_json(filepath, data):
    """Save JSON file with numpy support"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        return False

def crop_face(frame, bbox, margin=0.2):
    """
    Crop face from frame with margin
    """
    x, y, w, h = bbox
    
    # Add margin
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(frame.shape[1] - x, w + 2 * margin_x)
    h = min(frame.shape[0] - y, h + 2 * margin_y)
    
    return frame[y:y+h, x:x+w]

def resize_face(face_img, size=(160, 160)):
    """
    Resize face image to standard size
    """
    if face_img.size == 0:
        return face_img
    
    return cv2.resize(face_img, size)

def normalize_image(img):
    """
    Normalize image for model input
    """
    if img is None:
        return None
    
    # Convert to float and normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    
    # Standardize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
    
    return img

def calculate_fps(start_time, frame_count):
    """
    Calculate FPS
    """
    elapsed = time.time() - start_time
    if elapsed > 0:
        return frame_count / elapsed
    return 0

def create_video_writer(filename, fps, frame_size):
    """
    Create video writer for saving videos
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)