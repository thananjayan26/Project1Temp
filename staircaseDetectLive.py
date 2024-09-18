#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:13:35 2024

@author: thanan
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

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

# Start capturing video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Predict using the model
    prediction = model.predict(preprocessed_frame)
    confidence = np.max(prediction) * 100  # Convert to percentage
    predicted_class = class_names[np.argmax(prediction)]

    # Only display if confidence is above 80%
    if confidence >= 95:
        # Draw a rectangle at the center of the frame
        height, width, _ = frame.shape
        x1, y1 = int(width * 0.25), int(height * 0.25)  # Top-left corner of the box
        x2, y2 = int(width * 0.75), int(height * 0.75)  # Bottom-right corner of the box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        
        # Display the prediction and confidence on the frame
        text = f'{predicted_class} ({confidence:.2f}%)'
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with prediction
    cv2.imshow('Staircase Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
