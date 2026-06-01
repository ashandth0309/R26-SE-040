import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import json

train_dir = "faces_dataset/train"
test_dir = "faces_dataset/test"

IMG_SIZE = (160, 160)
BATCH_SIZE = 8

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
).flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_gen = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

num_classes = len(train_gen.class_indices)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/face_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

print("🚀 Training started...")

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=40,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

model.save("models/face_model.keras")

with open("models/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("✅ Model saved!")
print("📌 Class mapping:", train_gen.class_indices)