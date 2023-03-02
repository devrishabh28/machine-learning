import cv2
import face_recognition
import numpy as np

# Load the known images and their names
person_1_image = face_recognition.load_image_file("rishabh.jpg")
person_1_face_encoding = face_recognition.face_encodings(person_1_image)[0]
person_1_name = "Rishabh"

person_2_image = face_recognition.load_image_file("person_2.jpg")
person_2_face_encoding = face_recognition.face_encodings(person_2_image)[0]
person_2_name = "Person 2"

known_face_encodings = [
    person_1_face_encoding,
    person_2_face_encoding
]

known_face_names = [
    person_1_name,
    person_2_name
]

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video feed
    ret, frame = video_capture.read()

    # Convert the frame from BGR color to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Detect the faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with the known face encodings
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)

        name = "Unknown"

        # Check if any of the known face encodings match the detected face encoding
        if True in matches:
            # Find the index of the first match
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face and label it with the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 25),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()
