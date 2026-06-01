import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/emotion_model.h5")

class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48,48))
    face = face / 255.0
    face = face.reshape(1,48,48,1)

    pred = model.predict(face)
    emotion = class_names[np.argmax(pred)]

    cv2.putText(frame, emotion, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("AI Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()