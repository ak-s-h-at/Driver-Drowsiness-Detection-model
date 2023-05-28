import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound
import time

# Load the pre-trained drowsiness detection model
model = load_model('drowsiness_model.h5')

# Define the classes
classes = ['Closed Eyes', 'Open Eyes']

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to detect drowsiness in real-time
def detect_drowsiness():
    global eyes_closed, closed_start_time, alarm_start_time, cap  # Declare as global variables
    cap = cv2.VideoCapture(0)  # Use the default webcam (change index if necessary)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            eyes_closed = False

        for (x, y, w, h) in faces:
            eye_region = gray[y:y+h, x:x+w]

            eye_region = cv2.resize(eye_region, (60, 40))
            eye_region = eye_region.reshape((1, 40, 60, 1))
            eye_region = eye_region / 255.0

            prediction = model.predict(eye_region)
            label = classes[np.argmax(prediction)]

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if label == 'Closed Eyes':
                if not eyes_closed:
                    closed_start_time = time.time()
                    eyes_closed = True
                    alarm_start_time = None  # Reset alarm start time
                else:
                    elapsed_time = time.time() - closed_start_time
                    if elapsed_time >= 1:  # Eyes closed for 1 seconds
                        if alarm_start_time is None:
                            playsound('D:\Drowsiness Alert System\emergency-alarm-with-reverb-29431.mp3')
                            alarm_start_time = time.time()  # Start tracking alarm start time
                            alarm_duration = 2  # Set alarm duration to 2 seconds
                        else:
                            alarm_elapsed_time = time.time() - alarm_start_time
                            if alarm_elapsed_time >= alarm_duration:
                                playsound('D:\Drowsiness Alert System\emergency-alarm-with-reverb-29431.mp3')
                                eyes_closed = False
                                alarm_start_time = None
            else:
                eyes_closed = False
                alarm_start_time = None

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Initialize the global variables
eyes_closed = False
closed_start_time = 0
alarm_start_time = None
alarm_duration = 2  # Set the alarm duration in seconds

# Run the drowsiness detection
detect_drowsiness()

