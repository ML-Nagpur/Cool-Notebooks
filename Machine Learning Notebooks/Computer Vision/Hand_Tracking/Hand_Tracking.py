import cv2
import mediapipe as mp
import time

# Initialize video capture from the default camera (0)
cap = cv2.VideoCapture(0)

# Initialize Mediapipe modules for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialize variables for frame time tracking
pTime = 0
cTime = 0

# Main loop for capturing and processing video frames
while True:
    # Read a frame from the video capture
    success, img = cap.read()

    # Convert the BGR image to RGB format (required by Mediapipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with hand tracking
    results = hands.process(imgRGB)

    # Uncomment the line below to print hand landmarks data
    # print(results.multi_hand_landmarks)

    # Check if any hands are detected and draw landmarks and connections
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # Get the pixel coordinates of the landmark
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Print the landmark id and coordinates
                print(id, cx, cy)

                # Draw a circle around the landmark (e.g., fingertip)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Draw landmarks and connections on the hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate and display the frames per second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display the FPS on the image
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (25, 0, 255), 3)

    # Display the image in a window named 'Image'
    cv2.imshow("Image", img)

    # Check for 'q' key press to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
