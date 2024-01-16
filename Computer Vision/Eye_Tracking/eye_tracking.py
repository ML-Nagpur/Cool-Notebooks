# Make sure to download the pre-trained shape predictor model file (shape_predictor_68_face_landmarks.dat) 
# from the Dlib website (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and 
# provide the correct path to the file in the predictor_path variable.

import cv2
import dlib

# Load the pre-trained facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Open a video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Dlib to detect facial landmarks
    faces = dlib.get_frontal_face_detector()(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Increase box size around the eyes
        left_eye = (landmarks.part(36).x - 15, landmarks.part(36).y - 15,
                    landmarks.part(39).x + 15, landmarks.part(39).y + 15)
        right_eye = (landmarks.part(42).x - 15, landmarks.part(42).y - 15,
                     landmarks.part(45).x + 15, landmarks.part(45).y + 15)

        # Draw rectangles around the eyes
        cv2.rectangle(frame, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), (255, 0, 0), 2)
        cv2.rectangle(frame, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Eye Tracking', frame)

    # Break the loop if the 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
