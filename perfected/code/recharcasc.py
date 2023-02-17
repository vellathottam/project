import cv2
import pickle

# Load the known face encodings and names from the pickle file
with open("person.pickle", "rb") as handle:
    data, known_face_names = pickle.load(handle)
    known_face_encodings = data["encodings"]
    known_face_locations = data["locations"]

# Initialize the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start capturing video from the default camera
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_image = frame[y:y+h, x:x+w]

        # Convert the face image to RGB for compatibility with the face_recognition library
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Encode the face image using the face_recognition library
        face_encodings_in_image = face_recognition.face_encodings(rgb_face_image)

        # Loop over the known face encodings and compare with the encoding of the detected face
        for known_face_encoding, known_face_name in zip(known_face_encodings, known_face_names):
            matches = face_recognition.compare_faces(known_face_encoding, face_encodings_in_image)
            if True in matches:
                # The detected face matches with a known face, so display the name
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, known_face_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
