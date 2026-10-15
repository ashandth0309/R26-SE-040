import cv2
import numpy as np

class FallDetector:
    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence

    def process_frame(self, frame):
        try:
            height, width, _ = frame.shape
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            _, thresh = cv2.threshold(blurred, 75, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            is_fallen = False
            box = (0, 0, width, height)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Area romba perusa iruntha mattume (ignore small glares/half light)
                if area > 4000:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    box = (x, y, x + w, y + h)
                    
                    # 🛡️ STRICT CONDITION FOR FLOOR ONLY:
                    # 1. Width must be significantly greater than height (Horizontal layout)
                    # 2. Y-position must start below the middle (y > height * 0.35) so upper body lights won't trigger!
                    # 3. Bottom edge must be on the floor (y + h > height * 0.65)
                    if w > h * 1.35 and y > (height * 0.35) and (y + h) > (height * 0.65):
                        is_fallen = True

            return is_fallen, box

        except Exception as e:
            print(f"⚠️ FallDetector error: {e}")
            return False, (0, 0, 100, 100)