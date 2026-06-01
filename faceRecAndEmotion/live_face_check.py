import cv2
import numpy as np
import tensorflow as tf
import json

model = tf.keras.models.load_model("models/face_model.keras")

with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

IMG_SIZE = (160, 160)

CONFIDENCE_THRESHOLD = 0.80
MARGIN_THRESHOLD = 0.25

cap = cv2.VideoCapture(0)

print("🚀 Camera Started... Press q to quit")
print("📌 Classes:", class_names)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        try:
            padding = int(0.20 * w)

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size == 0:
                continue

            face = cv2.resize(face_roi, IMG_SIZE)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face, verbose=0)[0]

            index = int(np.argmax(pred))
            confidence = float(pred[index])

            sorted_preds = np.sort(pred)
            second_best = float(sorted_preds[-2]) if len(sorted_preds) > 1 else 0.0
            margin = confidence - second_best

            if confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
                label = "UNKNOWN"
                color = (0, 0, 255)
            else:
                label = class_names[index]
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        except Exception as e:
            print("Error:", e)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("🛑 Camera Stopped")