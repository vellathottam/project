import face_recognition
import pickle

# Create an empty list to store the names
names = []

# List to store the face locations of all the individuals
face_locations = []

# List to store the face encodings of all the individuals
face_encodings = []

# List to store the names of all the individuals
known_face_names = []

# Open the file containing the names
with open(r'C:\Users\donve\Desktop\project\perfected\docs\names.txt', 'r') as file:
    # Read the lines from the file
    lines = file.readlines()

# Loop over the lines in the file
for line in lines:
    # Remove the newline character from the end of the line
    line = line.strip()
    # Add the name to the list
    names.append(line)

# Load the images of known individuals
for name in names:
    images = []
    for i in range(1, 3):
        image = face_recognition.load_image_file(f"C:\\Users\\donve\\Desktop\\project\\mode1\\{name}_{i}.jpg")
        images.append(image)

    # Create the face encodings and face locations from the images
    for image in images:
        face_locations_in_image = face_recognition.face_locations(image)
        face_encodings_in_image = face_recognition.face_encodings(image)

        face_locations.append(face_locations_in_image)
        face_encodings.append(face_encodings_in_image)
        known_face_names.append(name)

# Store the face encodings and face locations in a pickle file
data = {"encodings": face_encodings, "locations": face_locations}
with open("person.pickle", "wb") as handle:
    pickle.dump((data, known_face_names), handle)
