import cv2
import time
import math
import mediapipe as mp

class FallDetector:
    def __init__(self, fall_time_threshold=2.0):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,   # Athigapaduthapadugirathu (doll filter seiya)
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.fall_time_threshold = fall_time_threshold
        self.fall_start_time = None
        self.is_fallen = False

    def detect_fall(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        status_text = "Status: Normal"
        box_color = (0, 255, 0)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Keypoints eduthal
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]

            # 1. Doll Filter: Visibility check (Unmaiyaana manidhanuku idhu athigamaaga irukkum)
            avg_vis = (left_shoulder.visibility + right_shoulder.visibility + 
                       left_hip.visibility + right_hip.visibility + nose.visibility) / 5.0

            # 2. Doll Filter: Size check (Udal periyathaga irukka vendum, doll chinna size)
            xs = [lm.x * w for lm in landmarks if lm.visibility > 0.5]
            ys = [lm.y * h for lm in landmarks if lm.visibility > 0.5]

            is_valid_human = False
            if len(xs) > 8 and len(ys) > 8:
                box_w = max(xs) - min(xs)
                box_h = max(ys) - min(ys)
                # Frame-la kurainthathu 20% area-vathu unmaiyaana manidhar cover seivaar
                if (box_w * box_h) > (w * h * 0.08) and avg_vis > 0.65:
                    is_valid_human = True

            if is_valid_human:
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * h
                shoulder_x = (left_shoulder.x + right_shoulder.x) / 2 * w
                hip_y = (left_hip.y + right_hip.y) / 2 * h
                hip_x = (left_hip.x + right_hip.x) / 2 * w

                y_diff = abs(shoulder_y - hip_y)
                x_diff = abs(shoulder_x - hip_x)
                angle = math.degrees(math.atan2(y_diff, x_diff))

                self.mp_draw.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )

                if angle < 45 or y_diff < (x_diff * 0.7):
                    if self.fall_start_time is None:
                        self.fall_start_time = time.time()
                    
                    elapsed = time.time() - self.fall_start_time
                    if elapsed >= self.fall_time_threshold:
                        self.is_fallen = True
                        status_text = "ALERT: PERSON FALLEN / UNCONSCIOUS!"
                        box_color = (0, 0, 255)
                    else:
                        status_text = f"Warning: Falling detected ({elapsed:.1f}s)"
                        box_color = (0, 165, 255)
                else:
                    self.fall_start_time = None
                    self.is_fallen = False
                    status_text = "Status: Standing / Normal"
                    box_color = (0, 255, 0)
            else:
                # Doll alladhu romba thoorathil ullavatrai ignore seigirom
                self.fall_start_time = None
                self.is_fallen = False
                status_text = "Status: Small Object/Doll Ignored"
                box_color = (180, 180, 180)

        else:
            self.fall_start_time = None
            self.is_fallen = False
            status_text = "Status: No Person"

        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        return frame, self.is_fallen