import cv2
import face_recognition

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(r'C:\Users\donve\Desktop\project\mode1\haarcascade_frontalface_default.xml')

# Load the face encodings of known individuals
known_face_encodings = []
known_face_names = []

# Open the file containing the names
with open(r'C:\Users\donve\Desktop\project\mode1\names.txt', 'r') as file:
    # Read the lines from the file
    lines = file.readlines()

# Create an empty list to store the names
names = []

# Loop over the lines in the file
for line in lines:
    # Remove the newline character from the end of the line
    line = line.strip()
    # Add the name to the list
    names.append(line)

# Load the images of known individuals
for name in names:
    image = face_recognition.load_image_file(f"C:\\Users\\donve\\Desktop\\project\\mode1\\{name}.jpg")
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert the frame to a format suitable for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find the face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop over the face encodings
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face encoding matches any known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match is found, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Live People Recognition', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
