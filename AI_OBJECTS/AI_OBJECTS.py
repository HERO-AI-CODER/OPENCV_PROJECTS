import torch

# Disable MKLDNN/OneDNN CPU optimizations
torch.backends.mkldnn.enabled = False

from ultralytics import YOLO

import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Robot vision started...")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera")
        break

    # Run object detection
    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confidence
            confidence = float(box.conf[0])

            # Class ID and name
            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            # Decide whether prediction is "exact"
            if confidence >= 0.75:
                color = (0, 255, 0)       # GREEN
                status = "EXACT"
            else:
                color = (0, 0, 255)       # RED
                status = "GUESS"

            # Draw box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # Label
            label = f"{status}: {object_name} {confidence * 100:.1f}%"

            # Label background
            (w, h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv2.rectangle(
                frame,
                (x1, y1 - h - 10),
                (x1 + w + 10, y1),
                color,
                -1
            )

            # Label text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # Robot status
    cv2.putText(
        frame,
        "ROBOT VISION ACTIVE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # Display
    cv2.imshow("Robot Object Identification", frame)

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()