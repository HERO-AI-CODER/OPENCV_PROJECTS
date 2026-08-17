import cv2
import mediapipe as mp

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Camera
cap = cv2.VideoCapture(0)

switch = False

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        # Count fingers
        fingers = 0

        if lm[8].y < lm[6].y:
            fingers += 1

        if lm[12].y < lm[10].y:
            fingers += 1

        if lm[16].y < lm[14].y:
            fingers += 1

        if lm[20].y < lm[18].y:
            fingers += 1

        # Thumb
        if lm[4].x < lm[3].x:
            fingers += 1

        # -----------------------
        # SWITCH CONTROL
        # -----------------------

        if fingers >= 3:
            switch = True

        elif fingers <= 1:
            switch = False

        # Draw hand
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        cv2.putText(
            frame,
            f"Fingers: {fingers}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    # Display switch state
    state = "ON" if switch else "OFF"

    cv2.putText(
        frame,
        f"SWITCH: {state}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0) if switch else (0, 0, 255),
        3
    )

    cv2.imshow("Hand Gesture Switch", frame)

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()