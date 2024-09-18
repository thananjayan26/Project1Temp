import numpy as np
import cv2
from scipy.spatial import distance as dist

# Constants
Known_distance = 60  # Inches
Known_width = 2.7  # Inches
thres = 0.5  # Detection threshold
nms_threshold = 0.2  # Non-Maximum Suppression threshold

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
ORANGE = (0, 69, 255)

# Camera object and model initialization
cap = cv2.VideoCapture(0)

# Load the face detection model (Ensure this file path is correct)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load object class names
classNames = []
with open('coco.names', 'r') as f:
    classNames = f.read().splitlines()

Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# Load the object detection model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Focal length finder function
def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

# Distance estimation function
def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
    distance = (real_object_width * Focal_Length) / object_width_in_frame
    return distance

# Face detection function
def face_data(image, CallOut, Distance_level):
    face_width = 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        face_width = w
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y

    return face_width, faces, face_center_x, face_center_y

# Load a reference image to calculate the focal length
ref_image = cv2.imread("lena.png")
ref_image_face_width, _, _, _ = face_data(ref_image, False, 0)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
print("Focal Length:", Focal_length_found)

while True:
    _, frame = cap.read()
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    
    if len(classIds) != 0:
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        for i in indices.flatten():
            box = bbox[i]
            confidence = str(round(confs[i][0], 2))
            color = Colors[classIds[i][0] - 1]
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(frame, classNames[classIds[i][0] - 1] + " " + confidence, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate distance
            distance = Distance_finder(Focal_length_found, Known_width, w)
            cv2.putText(frame, f"Distance: {round(distance, 2)} Inches", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    # Show the output frame
    cv2.imshow('Output', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
