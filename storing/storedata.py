import cv2
import face_recognition
import datetime
import csv

# Load the known faces and names from a CSV file
known_faces = []
known_names = []
with open('known_faces.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        img = face_recognition.load_image_file(row[0])
        encoding = face_recognition.face_encodings(img)[0]
        known_faces.append(encoding)
        known_names.append(row[1])

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)

# Loop over frames from the video file stream
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        # If a match was found in known_faces, just use the first one.
        # If a match was not found, set the name to "Unknown"
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Add the name to the list of names
        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Log the recognition data in a CSV file
        with open('recognition_log.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, top, right, bottom, left])

    # Increment the frame counter
    frame_number += 1

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
video_capture.release()
cv2.destroyAllWindows()
