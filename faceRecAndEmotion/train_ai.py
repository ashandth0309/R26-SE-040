import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ==============================
# 1. Dataset Paths
# ==============================
train_dir = 'data/train'
test_dir = 'data/test'

# ==============================
# 2. Data Preprocessing
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# ==============================
# 3. CNN Model (Improved)
# ==============================
model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(7, activation='softmax')
])

# ==============================
# 4. Compile Model
# ==============================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 5. Train Model
# ==============================
print("🚀 Training started...")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=25
)

# ==============================
# 6. Evaluate Model
# ==============================
print("\n📊 Evaluating model on test data...")
test_loss, test_acc = model.evaluate(test_generator)

print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

# ==============================
# 7. Save Model
# ==============================
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/emotion_model.h5')

print("\n✅ Model saved successfully!")

# ==============================
# 8. Save Class Labels
# ==============================
class_labels = train_generator.class_indices
print("\n📌 Class Mapping:", class_labels)

with open("models/class_labels.txt", "w") as f:
    f.write(str(class_labels))

print("✅ Class labels saved!")