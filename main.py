import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

index_x = 0
index_y = 0
middle_x = 0
middle_y = 0
thumb_x = 0
thumb_y = 0

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:

        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = (landmark.x*frame_width)
                y = (landmark.y*frame_height)

                if id == 8:
                    index_x = screen_width / frame_width * x
                    index_y = screen_width / frame_width * y
                    pyautogui.moveTo(index_x, index_y)

                if id == 12:
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_width / frame_width * y

                if id == 4:
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_width / frame_width * y
                    if (abs(middle_y - thumb_y)) < 50:
                        pyautogui.click()
                        pyautogui.sleep(1)

                if id == 16:
                    ring_x = screen_width / frame_width * x
                    ring_y = screen_width / frame_width * y

                if id == 20:
                    pinky_x = screen_width / frame_width * x
                    pinky_y = screen_width / frame_width * y

                if id == 0:
                    wrist_x = screen_width / frame_width * x
                    wrist_y = screen_width / frame_width * y
                    
    cv2.waitKey(1)
