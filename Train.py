import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
from playsound import playsound

# Path to the dataset folders
closed_eyes_dir = "D:\Drowsiness Alert System\Dataset\closedeyes"
open_eyes_dir = "D:\Drowsiness Alert System\Dataset\openeyes"

def load_dataset():
    # Supported image file extensions
    valid_extensions = ['.jpg', '.jpeg', '.png']

    # Load closed eyes images
    closed_eyes_images = []
    for file in os.listdir(closed_eyes_dir):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(closed_eyes_dir, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (60, 40))
                closed_eyes_images.append(img)

    # Load open eyes images
    open_eyes_images = []
    for file in os.listdir(open_eyes_dir):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(open_eyes_dir, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (60, 40))
                open_eyes_images.append(img)

    # Check if any images were loaded
    if len(closed_eyes_images) == 0 or len(open_eyes_images) == 0:
        raise ValueError("No images found in the dataset folders.")

    # Create the labels
    closed_eyes_labels = [0] * len(closed_eyes_images)
    open_eyes_labels = [1] * len(open_eyes_images)

    # Combine the images and labels
    images = closed_eyes_images + open_eyes_images
    labels = closed_eyes_labels + open_eyes_labels

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Reshape images to include the channel dimension
    images = images.reshape(images.shape + (1,))

    # Normalize the image data
    images = images / 255.0

    # Shuffle the data
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    # Convert labels to categorical
    labels = to_categorical(labels)

    return images, labels


# Preprocess the image (resize, crop, etc.)
def preprocess_image(img):
    # Resize the image to a consistent size
    img = cv2.resize(img, (60, 40))
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Reshape the image to include the channel dimension
    img = img.reshape((img.shape[0], img.shape[1], 1))
    # Normalize the image data
    img = img / 255.0
    return img


# Load the dataset
images, labels = load_dataset()

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=train_images[0].shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# Save the trained model
model.save('drowsiness_model.h5')

print("Model trained and saved successfully.")

