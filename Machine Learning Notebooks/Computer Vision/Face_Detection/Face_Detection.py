import cv2
import mediapipe as mp
import time

# Initialize the video capture
cap = cv2.VideoCapture(0)
pTime = 0

# Initialize Mediapipe modules for face detection
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    # Read a frame from the video capture
    success, img = cap.read()

    # Convert the image to RGB format (required by Mediapipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with face detection
    results = faceDetection.process(imgRGB)
    print(results)

    # Draw bounding boxes and labels on the image if faces are detected
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 5)

    # Calculate and display the frames per second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # Display the image in a window named 'Image'
    cv2.imshow('Image', img)

    # Check for 'q' key press to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()