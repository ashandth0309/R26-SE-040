import cv2
import numpy as np
import tensorflow as tf

# ======================
# Load model
# ======================
model = tf.keras.models.load_model("models/face_model.h5")

# IMPORTANT: match training folder order
class_names = ["ashandth", "sobiya"]

# ======================
# Face detector
# ======================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face_roi = gray[y:y+h, x:x+w]

        # preprocess (MATCH TRAIN SIZE)
        face = cv2.resize(face_roi, (100, 100))
        face = face / 255.0
        face = face.reshape(1, 100, 100, 1)

        # prediction
        pred = model.predict(face, verbose=0)

        index = np.argmax(pred)
        confidence = np.max(pred)

        # ======================
        # UNKNOWN FIX (IMPORTANT)
        # ======================
        if confidence < 0.70:
            label = "UNKNOWN"
        else:
            label = class_names[index]

        # draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Face Recognition AI Robot", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()