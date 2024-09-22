#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:05:49 2024

@author: thanan
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from playsound import playsound  # Import playsound for audio output
import pyttsx3


# Load the trained model
model = tf.keras.models.load_model('staircase_detection_model.h5')

# Class names
class_names = ['Ascending Stairs', 'Descending Stairs']

# Initialize video capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Function to preprocess the frame
def preprocess_frame(frame):
    # Resize frame to the size expected by the model (128x128)
    resized_frame = cv2.resize(frame, (128, 128))
    # Convert frame to array and normalize pixel values
    img_array = image.img_to_array(resized_frame) / 255.0
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Flag to prevent repeating the warning sound
played_warning = False


# Constants for object detection
thres = 0.5  # Threshold to detect object
nms_threshold = 0.2  # Non-maxima suppression threshold
Known_width = 7  # Width in inches for the reference object (person)

# Dynamic known distances for each class (in inches)
known_distances = {
    'person': 150, 'bicycle': 120, 'car': 300, 'motorcycle': 180, 'airplane': 1000,
    'bus': 400, 'train': 800, 'truck': 500, 'boat': 600, 'traffic light': 200,'fire hydrant': 50, 'street sign': 150, 'stop sign': 150, 'parking meter': 100, 'bench': 100,
    'bird': 30, 'cat': 40, 'dog': 60, 'horse': 150, 'sheep': 100, 'cow': 200,
    'elephant': 400, 'bear': 250, 'zebra': 150, 'giraffe': 300, 'hat': 20, 'backpack': 30,
    'umbrella': 40, 'shoe': 20, 'eye glasses': 10, 'handbag': 30, 'tie': 10, 'suitcase': 50,
    'frisbee': 20, 'skis': 100, 'snowboard': 100, 'sports ball': 15, 'kite': 100,
    'baseball bat': 40, 'baseball glove': 20, 'skateboard': 50, 'surfboard': 80, 'tennis racket': 30,
    'bottle': 10, 'plate': 15, 'wine glass': 8, 'cup': 10, 'fork': 7, 'knife': 7, 'spoon': 7,
    'bowl': 15, 'banana': 8, 'apple': 8, 'sandwich': 10, 'orange': 8, 'broccoli': 8, 'carrot': 8,
    'hot dog': 10, 'pizza': 12, 'donut': 10, 'cake': 20, 'chair': 100, 'couch': 200, 'potted plant': 30,
    'bed': 250, 'mirror': 70, 'dining table': 200, 'window': 150, 'desk': 150, 'toilet': 60,
    'door': 200, 'tv': 100, 'laptop': 20, 'mouse': 5, 'remote': 7, 'keyboard': 15, 'cell phone': 1,
    'microwave': 40, 'oven': 60, 'toaster': 20, 'sink': 50, 'refrigerator': 100, 'blender': 20,
    'book': 10, 'clock': 15, 'vase': 30, 'scissors': 5, 'teddy bear': 20, 'hair drier': 15, 'toothbrush': 5, 'hair brush': 7
    # Add all remaining distances here
}

# Colors (BGR Format)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
font = cv2.FONT_HERSHEY_PLAIN

# Camera Object
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load COCO class names
classNames = []
with open('coco.names', 'r') as f:
    classNames = f.read().splitlines()

# Random color for each class
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# Load neural network
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Function to find focal length
def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

# Distance estimation function
def Distance_finder(Focal_Length, real_object_width, width_in_frame):
    distance = (real_object_width * Focal_Length) / width_in_frame
    return distance

# Detect face function
def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        face_width = w
    return face_width

# Reading reference image from directory
ref_image = cv2.imread("lena.png")
ref_image_face_width = face_data(ref_image)

# Compute focal length using the reference object (person) known width and distance
Focal_length_found = FocalLength(known_distances['person'], Known_width, ref_image_face_width)

closest_object_section = None




while True:
    _, frame = cap.read()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # Draw section lines
    section_width = frame_width // 4
    cv2.line(frame, (section_width, 0), (section_width, frame_height), (255, 255, 0), 2)
    cv2.line(frame, (2 * section_width, 0), (2 * section_width, frame_height), (255, 255, 0), 2)
    cv2.line(frame, (3 * section_width, 0), (3 * section_width, frame_height), (255, 255, 0), 2)

    # Object detection
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # Non-max suppression and initialization of closest object variables
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    closest_object_distance = float('inf')
    closest_object_position = ""

    # Ensure indices is iterable
    if len(classIds) != 0 and len(indices) > 0:  # Check that indices is not empty
        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i  # Ensure i is a scalar index
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            object_name = classNames[classIds[i] - 1]
            color = Colors[classIds[i] - 1]

            # Use dynamic known distance for the detected object
            if object_name in known_distances:
                object_known_width = Known_width  # Width in inches for the reference object
                object_known_distance = known_distances[object_name]

                # Calculate the distance of the detected object from the camera
                distance = Distance_finder(Focal_length_found, object_known_width, w)
                distance = round(distance, 2)

                # Determine the object's section (left, slightly left, slightly right, right)
                object_center_x = x + w / 2

                if object_center_x < frame_width / 4:
                    section = "Left"
                elif object_center_x < frame_width / 2:
                    section = "Slightly Left"
                elif object_center_x < 3 * frame_width / 4:
                    section = "Slightly Right"
                else:
                    section = "Right"

                # Check if this object is the closest one
                if distance < closest_object_distance:
                    closest_object_distance = distance
                    closest_object_position = section

                # Draw bounding box and distance
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{object_name} {distance} Inches", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # If closest object section changes, announce it
    if closest_object_position and closest_object_position != closest_object_section:
        closest_object_section = closest_object_position
        engine.say(f"The closest object is in the {closest_object_section}")
        engine.runAndWait()  # Wait for speech to complete

    # Display the frame
    cv2.imshow('Object and Face Distance Estimation', frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
