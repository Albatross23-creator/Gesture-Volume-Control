import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np
import math

# Initialize webcam
cap = cv2.VideoCapture(0)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Access system volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()  # (-65.25, 0.0)
min_vol = vol_range[0]
max_vol = vol_range[1]
vol = 0
vol_bar = 400
vol_perc = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            lm_list = hand_landmark.landmark

            # Index finger tip (id=8), thumb tip (id=4)
            x1 = int(lm_list[4].x * w)
            y1 = int(lm_list[4].y * h)
            x2 = int(lm_list[8].x * w)
            y2 = int(lm_list[8].y * h)

            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Calculate distance
            length = math.hypot(x2 - x1, y2 - y1)

            # Convert distance to volume level
            vol = np.interp(length, [20, 200], [min_vol, max_vol])
            vol_bar = np.interp(length, [20, 200], [400, 150])
            vol_perc = np.interp(length, [20, 200], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    # Draw volume bar and percentage
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_perc)} %', (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
