import cv2
import mediapipe as mp
import pyautogui

# Initialize video capture and mediapipe hand detector
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

def convert_coordinates(x, y, frame_width, frame_height):
    screen_x = screen_width / frame_width * x
    screen_y = screen_height / frame_height * y
    return screen_x, screen_y

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            index_coords = convert_coordinates(landmarks[8].x * frame_width, landmarks[8].y * frame_height, frame_width, frame_height)
            pyautogui.moveTo(*index_coords)

            middle_coords = convert_coordinates(landmarks[12].x * frame_width, landmarks[12].y * frame_height, frame_width, frame_height)
            thumb_coords = convert_coordinates(landmarks[4].x * frame_width, landmarks[4].y * frame_height, frame_width, frame_height)

            # Double click if thumb and middle finger are close
            if abs(middle_coords[1] - thumb_coords[1]) < 50:
                pyautogui.doubleClick(x=thumb_coords[0], y=middle_coords[1])
                pyautogui.sleep(1)

            # Left click if thumb and index finger are close
            if abs(index_coords[1] - thumb_coords[1]) < 50:
                (pyautogui.click(x=thumb_coords[0], y=index_coords[1]))
                pyautogui.sleep(1)

            # Right click if index and middle finger are close 
            if abs(middle_coords[1] - index_coords[1]) < 50:
                pyautogui.rightClick(x=index_coords[0], y=middle_coords[1])
                pyautogui.sleep(1)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
