import cv2
import mediapipe as mp
import pyautogui
import os
import warnings

# Disable XNNPACK for TensorFlow Lite to avoid warnings related to feedback manager
os.environ['TF_LITE_USE_XNNPACK'] = '0'

# Suppress deprecated warnings from protobuf
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Initialize variables
x1 = y1 = x2 = y2 = 0
webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

while True:
    # Capture frame from webcam
    _, image = webcam.read()
    image = cv2.flip(image, 1)  # Flip the image horizontally
    frame_height, frame_width, _ = image.shape

    # Convert image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    # If hands are detected
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark
            x1 = y1 = x2 = y2 = 0
            x3 = y3 = x4 = y4 = 0

            # Detect landmarks for different fingers
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Tip of the index finger
                    cv2.circle(img=image, center=(x, y), radius=15, color=(0, 255, 255), thickness=3)
                    x1, y1 = x, y

                if id == 4:  # Tip of the thumb
                    cv2.circle(img=image, center=(x, y), radius=15, color=(0, 255, 255), thickness=3)
                    x2, y2 = x, y

                if id == 12:  # Tip of the middle finger
                    cv2.circle(img=image, center=(x, y), radius=15, color=(255, 0, 0), thickness=3)
                    x3, y3 = x, y

                if id == 20:  # Tip of the pinky finger
                    cv2.circle(img=image, center=(x, y), radius=15, color=(255, 0, 0), thickness=3)
                    x4, y4 = x, y

            # Calculate distance between index finger and thumb (for volume control)
            dist_thumb_index = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # Pause/Play Gesture (Pinch - index and thumb tips close together)
            if dist_thumb_index < 50:
                cv2.putText(image, "Pause/Play", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pyautogui.press("playpause")

            # Skip Gesture (Pinky finger far to the right of middle finger)
            if x4 > x3 + 100:  # Adjust threshold for "rightward" gesture
                cv2.putText(image, "Skip", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pyautogui.press("nexttrack")

            # Volume Control (based on distance between thumb and index finger)
            if dist_thumb_index > 50:
                pyautogui.press("volumeup")
            elif dist_thumb_index < 30:
                pyautogui.press("volumedown")

    # Display the processed image
    cv2.imshow("Gesture Volume Control", image)

    # Exit on pressing 'ESC' key
    key = cv2.waitKey(10)
    if key == 27:  # Escape key to exit
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()