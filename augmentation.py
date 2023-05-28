import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Path to the directory containing closed eye images
closed_eye_dir = 'D:\Drowsiness Alert System\Dataset\closedeyes'

# Output directory to save augmented images
output_dir = 'D:\Drowsiness Alert System\Dataset\close2'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load closed eye images
closed_eye_images = []
for filename in os.listdir(closed_eye_dir):
    image_path = os.path.join(closed_eye_dir, filename)
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (100, 100))  # Resize the image to a common size
    closed_eye_images.append(resized_image)

# Convert the images to a NumPy array
closed_eye_images = np.array(closed_eye_images)

# Create an instance of the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Random rotation between -20 and +20 degrees
    width_shift_range=0.1,  # Random horizontal shift
    height_shift_range=0.1,  # Random vertical shift
    shear_range=0.2,  # Random shear
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill mode for filling in newly created pixels
)

# Perform data augmentation and save the augmented images
num_augmented_images = 0
for i, image in enumerate(closed_eye_images):
    image = image.reshape((1,) + image.shape)  # Reshape to (1, height, width, channels)
    save_prefix = 'closed_eye_augmented_{}'.format(i)

    # Generate augmented images
    aug_counter = 0  # Counter to track the number of augmented images per input image
    for batch in datagen.flow(image, batch_size=1, save_to_dir=output_dir,
                              save_prefix=save_prefix, save_format='png'):
        num_augmented_images += 1
        aug_counter += 1

        if aug_counter >= 10:  # Limit the number of augmented images per input image to 10
            break

    if num_augmented_images >= 156:  # Stop generating augmented images once 156 images are reached
        break

print("Data augmentation complete.")
