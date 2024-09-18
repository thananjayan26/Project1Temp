import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()

# Corrected output layers extraction
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the camera's focal length and known width for calibration
KNOWN_WIDTH = 14.3  # Example: known width of a reference object in cm
FOCAL_LENGTH = 615  # Example: estimated focal length of the camera

# Initialize the video capture
cap = cv2.VideoCapture(0)

def calculate_distance(knownWidth, focalLength, perWidth):
    """ Calculate distance from camera to object. """
    return (knownWidth * focalLength) / perWidth

while cap.isOpened():
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Detecting objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences = [], []

    # Loop over each detection
    for out in detections:
        for detection in out:
            scores = detection[5:]
            confidence = max(scores)

            if confidence > 0.5:
                # Object detected
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if any indices were returned and handle the single detection case
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
        for i in indices:
            x, y, w, h = boxes[i]

            # Calculate the distance to the detected object
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, w)

            # Draw the bounding box and distance
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{int(distance)} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    elif isinstance(indices, tuple) and len(indices) > 0:
        for i in indices[0]:
            i = int(i)  # Convert to integer if necessary
            x, y, w, h = boxes[i]

            # Calculate the distance to the detected object
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, w)

            # Draw the bounding box and distance
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{int(distance)} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Distance Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
