import face_recognition
import pickle

# Load the images of the person
images = []
for i in range(1, 6):
    image = face_recognition.load_image_file(f"person_{i}.jpg")
    images.append(image)

# Create the face encodings and face locations from the images
face_encodings = []
face_locations = []
for image in images:
    face_encodings_in_image = face_recognition.face_encodings(image)
    face_locations_in_image = face_recognition.face_locations(image)

    face_encodings.extend(face_encodings_in_image)
    face_locations.extend(face_locations_in_image)

# Store the face encodings and face locations in a pickle file
data = {"encodings": face_encodings, "locations": face_locations}
with open("person.pickle", "wb") as handle:
    pickle.dump(data, handle)
