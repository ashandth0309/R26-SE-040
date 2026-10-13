import cv2
import mediapipe as mp
import numpy as np

class FallDetector:
    def __init__(self, min_confidence=0.5):
        # MediaPipe Pose initialize panrom
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
        # Skeleton drawing utilities-ai add panrom
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process_frame(self, frame):
        """
        Frame-ai analyze panni person vizhunthuttaraanu allathu mayanki padithirukkaannu check pannum.
        Returns: (is_fallen (bool), bounding_box (tuple or None))
        """
        if frame is None:
            return False, None
            
        height, width, _ = frame.shape
        # OpenCV BGR to RGB conversion
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            # 1. Frame-la red joints & white/colored skeleton lines-ai draw panrom
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3), # Red joints
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2) # Skeleton lines
            )
            
            landmarks = results.pose_landmarks.landmark
            
            # Keypoints-in X and Y coordinates-ai edukkrom
            x_coords = [lm.x * width for lm in landmarks]
            y_coords = [lm.y * height for lm in landmarks]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            box_width = max_x - min_x
            box_height = max_y - min_y
            
            # 2. Advanced Fall & Unconscious / Fainted Detection Logic:
            # - Condition A: Width height-ai vida athigamagirum (Horizontal posture / Keezha vizhunthal)
            # - Condition B: Head (Nose) matrum Hips ore horizontal line-kku varum (Mayanki padithirunthal diff romba kammiya irukkum)
            nose_y = landmarks[self.mp_pose.PoseLandmark.NOSE].y * height
            left_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * height
            right_hip_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * height
            avg_hip_y = (left_hip_y + right_hip_y) / 2
            
            if box_width > box_height * 1.1 or abs(nose_y - avg_hip_y) < 70:
                return True, (int(min_x), int(min_y), int(max_x), int(max_y))
                
        return False, None