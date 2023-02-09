import face_recognition
import pickle

# Load the image of the person
image = face_recognition.load_image_file("person1.jpg")

# Create the face encoding of the person
face_encoding = face_recognition.face_encodings(image)[0]

# Store the face encoding in a dictionary with the person's name as the key
person_encodings = {"person1": face_encoding}

# Serialize the dictionary and store it in a pickle file
with open("person_encodings.pickle", "wb") as f:
    pickle.dump(person_encodings, f)

# Load the pickle file
with open("person_encodings.pickle", "rb") as f:
    loaded_person_encodings = pickle.load(f)

print("Loaded person encodings:", loaded_person_encodings)
