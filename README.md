**Drowsiness Detection Model**
This repository contains a drowsiness detection model built using deep learning techniques. The model is capable of detecting whether a person's eyes are open or closed in real-time, making it useful for applications such as driver drowsiness monitoring or alert systems.

**Features**
    Utilizes a Convolutional Neural Network (CNN) architecture for eye state classification.
    Trained on a large dataset of labeled eye images, including both closed and open eyes.
    Achieves high accuracy in distinguishing between closed and open eyes in real-time video streams.
    Integrates with a webcam to capture live video and perform real-time drowsiness detection.
    Includes an alarm system that alerts the user when closed eyes are detected for a certain duration.
    Simple and lightweight implementation using Python and popular libraries like OpenCV, Keras, and playsound.

**Usage**
    Install the required dependencies by running pip install -r requirements.txt.
    Run the detect_drowsiness.py script to start the drowsiness detection system.
    Ensure a webcam is connected and functioning properly for live video capture.
    The system will analyze the video feed in real-time, detecting closed or open eyes.
    If closed eyes are detected for a specified duration, an alarm will be triggered.
    The alarm sound will play until the eyes are open again or a certain timeout is reached.
    
    Feel free to explore the code and customize it according to your specific requirements.

**Dataset**
The model was trained on a curated dataset of eye images, comprising both closed and open eyes. 

**License**
This project is licensed under the MIT License, allowing for unrestricted use, modification, and distribution.
