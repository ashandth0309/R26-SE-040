import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Define Dataset Paths
train_dir = 'data/train' # Path to your training data folder
test_dir = 'data/test'   # Path to your testing data folder

# 2. Data Preprocessing (Image Augmentation)
# Rescale normalizes pixel values from [0-255] to [0-1]
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directory and convert them to 48x48 grayscale
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(48, 48), 
    batch_size=64, 
    color_mode='grayscale', 
    class_mode='categorical'
)

# 3. Build the AI Model (Convolutional Neural Network - CNN)
model = Sequential([
    # First Convolutional layer to detect features
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    
    # Second Convolutional layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Flattening the 2D images into a 1D vector
    Flatten(),
    
    # Fully connected layers
    Dense(128, activation='relu'),
    
    # Output layer with 7 units (one for each emotion) using Softmax
    Dense(7, activation='softmax')
])

# Compile the model with Adam optimizer and Categorical Crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Start the Training Process
print("🚀 Training started... This may take 10-20 minutes depending on your PC.")
model.fit(train_generator, epochs=10)

# 5. Save the Trained Model
if not os.path.exists('models'): 
    os.makedirs('models')

# This saves the "brain" of your AI so you can use it in your robot
model.save('models/emotion_model.h5')
print("✅ Success! models/emotion_model.h5 has been created.")