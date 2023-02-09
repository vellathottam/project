import face_recognition
import pickle

# List to store the face encodings of all the individuals
known_face_encodings = []

# List to store the names of all the individuals
known_face_names = []

# Loop over the images of all the individuals
for name in ["person1", "person2", "person3"]:
    # Load the image of the individual
    image = face_recognition.load_image_file(f"{name}.jpg")

    # Get the face locations in the image
    face_locations = face_recognition.face_locations(image)

    # Get the face encodings in the image
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Add the face encodings of the individual to the list
    for face_encoding in face_encodings:
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

# Store the face encodings and names in a pickle file
with open("face_encodings.pkl", "wb") as f:
    pickle.dump((known_face_encodings, known_face_names), f)
