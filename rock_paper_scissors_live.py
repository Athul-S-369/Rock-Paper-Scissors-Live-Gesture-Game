import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Define gestures
GESTURES = {"rock": 0, "paper": 1, "scissors": 2}
GESTURE_NAMES = ["Rock", "Paper", "Scissors"]

# Animation settings
ANIMATION_DURATION = 1.5  # seconds
TIMER_BETWEEN_ROUNDS = 2  # seconds

# Smoothing settings
GESTURE_BUFFER_SIZE = 7
GESTURE_CONFIRM_COUNT = 5

def get_hand_gesture(hand_landmarks):
    # Get landmark coordinates
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks:
        lm = hand_landmarks.landmark
        # Thumb
        fingers.append(1 if lm[tips_ids[0]].x < lm[tips_ids[0] - 1].x else 0)
        # Other fingers
        for i in range(1, 5):
            fingers.append(1 if lm[tips_ids[i]].y < lm[tips_ids[i] - 2].y else 0)
        # Gesture logic
        if fingers == [0, 0, 0, 0, 0]:
            return "rock"
        elif fingers == [0, 1, 1, 1, 1]:
            return "paper"
        elif fingers == [0, 1, 1, 0, 0]:
            return "scissors"
    return None

def get_winner(user_move, computer_move):
    if user_move == computer_move:
        return "Draw"
    elif (user_move == "rock" and computer_move == "scissors") or \
         (user_move == "scissors" and computer_move == "paper") or \
         (user_move == "paper" and computer_move == "rock"):
        return "You Win!"
    else:
        return "Computer Wins!"

def draw_centered_text(frame, text, y, color=(255,255,255), scale=2, thickness=3):
    h, w, _ = frame.shape
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = (w - size[0]) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def most_common_gesture(buffer):
    if not buffer:
        return None
    counts = {g: buffer.count(g) for g in set(buffer) if g is not None}
    if not counts:
        return None
    gesture, count = max(counts.items(), key=lambda x: x[1])
    if count >= GESTURE_CONFIRM_COUNT:
        return gesture
    return None

def main():
    cap = cv2.VideoCapture(0)
    computer_move = random.choice(list(GESTURES.keys()))
    result = ""
    user_move = None
    round_state = "waiting"  # waiting, animating, showing_result, timer
    animation_start = 0
    timer_start = 0
    animation_move = 0
    gesture_buffer = deque(maxlen=GESTURE_BUFFER_SIZE)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_mp = hands.process(rgb)
        detected_move = None
        if result_mp.multi_hand_landmarks:
            for hand_landmarks in result_mp.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = get_hand_gesture(hand_landmarks)
                if gesture:
                    detected_move = gesture
        gesture_buffer.append(detected_move)
        smoothed_gesture = most_common_gesture(list(gesture_buffer))
        # State machine
        if round_state == "waiting":
            draw_centered_text(frame, "Show your move!", 100, (255,255,0), 1.5, 3)
            if smoothed_gesture:
                user_move = smoothed_gesture
                animation_start = time.time()
                round_state = "animating"
        elif round_state == "animating":
            elapsed = time.time() - animation_start
            # Animation: cycle through moves
            animation_move = int((elapsed / ANIMATION_DURATION) * 9) % 3
            draw_centered_text(frame, f"Computer is choosing...", 100, (0,255,255), 1.2, 2)
            draw_centered_text(frame, GESTURE_NAMES[animation_move], 200, (0,255,255), 2.5, 5)
            if elapsed >= ANIMATION_DURATION:
                computer_move = random.choice(list(GESTURES.keys()))
                result = get_winner(user_move, computer_move)
                round_state = "showing_result"
                timer_start = time.time()
        elif round_state == "showing_result":
            draw_centered_text(frame, f"You: {user_move.capitalize()}  |  Computer: {computer_move.capitalize()}", 100, (255,255,255), 1.2, 2)
            color = (0,255,0) if result=="You Win!" else (0,0,255) if result=="Computer Wins!" else (255,255,0)
            draw_centered_text(frame, result, 200, color, 2.5, 5)
            # Timer for next round
            seconds_left = TIMER_BETWEEN_ROUNDS - int(time.time() - timer_start)
            if seconds_left > 0:
                draw_centered_text(frame, f"Next round in {seconds_left}...", 300, (255,255,255), 1.2, 2)
            else:
                round_state = "waiting"
                user_move = None
                result = ""
        # Overlay info
        cv2.rectangle(frame, (0,0), (w, 80), (0,0,0), -1)
        cv2.putText(frame, f"Press ESC to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Rock Paper Scissors - Live", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 