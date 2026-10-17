import cv2
import time

from faceRecAndEmotion.fall_detection_module import FallDetector


def main():
    detector = FallDetector(min_confidence=0.5)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("BUDDY Fall Detection Test")
    print("-------------------------")
    print("Press Q to exit.")

    last_alert_time = 0.0
    alert_cooldown = 3.0

    # Keep visual alert visible after detection
    alert_hold_seconds = 3.0
    alert_visible_until = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera frame.")
            break

        is_fallen, box = detector.process_frame(frame)

        current_time = time.time()

        x1, y1, x2, y2 = box

        # If detector sees a fall, latch the alert
        if is_fallen:
            alert_visible_until = current_time + alert_hold_seconds

            if current_time - last_alert_time > alert_cooldown:
                print("WARNING: Possible fall detected.")
                last_alert_time = current_time

        # Alert remains visible for 3 seconds
        alert_active = current_time < alert_visible_until

        if x2 > x1 and y2 > y1:
            if alert_active:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

        if alert_active:
            cv2.rectangle(
                frame,
                (10, 10),
                (430, 65),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "POSSIBLE FALL DETECTED",
                (20, 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        else:
            cv2.putText(
                frame,
                "NORMAL / NO FALL",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.imshow("BUDDY - Fall Detection Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Fall detection test stopped.")


if __name__ == "__main__":
    main()
