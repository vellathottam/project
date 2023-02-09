import cv2
import os
import numpy as np
from PIL import Image

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the label encoder
labels = {}
current_id = 0

# Create the dataset directory
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Get all the person names in the dataset directory
person_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

# Iterate through each person name
for person_name in person_names:
    person_dir = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    # Assign an id to the person name
    labels[person_name] = current_id
    current_id += 1

    # Get all the images for the person
    images = [os.path.join(person_dir, i) for i in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, i))]

    # Iterate through each image
    for image_path in images:
        # Read the image
        image = Image.open(image_path).convert("L")
        image_array = np.array(image, "uint8")

        # Detect faces
        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5)

        # Iterate through each face
        for (x, y, w, h) in faces:
            roi = image_array[y:y+h, x:x+w]

            # Train the face recognizer
            recognizer.update([roi], [labels[person_name]])

# Save the face recognizer and label encoder
recognizer.write("face-trainner.yml")
with open("labels.pickle", "wb") as f:
    pickle.dump(labels, f)
