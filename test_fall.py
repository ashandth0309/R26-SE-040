import cv2
from faceRecAndEmotion.fall_detection_module import FallDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = FallDetector(fall_time_threshold=2.0)

    print("🚀 Fall Detection System Started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, fallen = detector.detect_fall(frame)
        if fallen:
            print("🚨 Alert: Person has fallen down!")

        cv2.imshow("BUDDY - Human Pose & Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()