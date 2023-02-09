import cv2

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")

# Load the label encoder
labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Recognize the face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:
            print(id_)
            print(labels[id_])
            name = labels[id_]
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    
    # Show the video frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
