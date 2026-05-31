import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

# ======================
# Dataset paths
# ======================
train_dir = "faces_dataset/train"
test_dir = "faces_dataset/test"

# ======================
# STRONG AUGMENTATION (IMPORTANT)
# ======================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True
).flow_from_directory(
    train_dir,
    target_size=(100,100),
    color_mode="grayscale",
    class_mode="categorical"
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=(100,100),
    color_mode="grayscale",
    class_mode="categorical"
)

# ======================
# MODEL (FIXED FOR SMALL DATASET)
# ======================
model = Sequential([

    Conv2D(32,(3,3),activation="relu",input_shape=(100,100,1)),
    MaxPooling2D(2,2),
    BatchNormalization(),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(2,2),
    BatchNormalization(),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(2,2),
    BatchNormalization(),

    Flatten(),

    Dense(64,activation="relu"),
    Dropout(0.5),

    Dense(len(train_gen.class_indices),activation="softmax")
])

# ======================
# COMPILE
# ======================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# EARLY STOPPING
# ======================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ======================
# TRAIN
# ======================
print("🚀 Training started...")
model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=25,
    callbacks=[early_stop]
)

# ======================
# SAVE MODEL
# ======================
os.makedirs("models", exist_ok=True)
model.save("models/face_model.h5")

print("✅ Model saved!")

# ======================
# CLASS MAPPING
# ======================
print("📌 Class mapping:", train_gen.class_indices)