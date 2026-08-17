import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================

# YOLO model
model = YOLO("yolo26n.pt")

# Haar Cascade for face detection
cascade_path = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("ERROR: Could not load Haar Cascade!")
    print("Make sure haarcascade_frontalface_default.xml")
    print("is inside the same folder as this Python file.")
    exit()

# Folder for captured faces
face_folder = os.path.join(
    os.path.dirname(__file__),
    "captured_faces"
)

os.makedirs(face_folder, exist_ok=True)

# Robot observation file
observation_file = os.path.join(
    os.path.dirname(__file__),
    "robot_observations.txt"
)

# Camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("=" * 60)
print("ROBOT VISION SYSTEM")
print("=" * 60)
print("Camera: ONLINE")
print("Object detection: ONLINE")
print("Face detection: ONLINE")
print("Face storage: ONLINE")
print("Press Q to stop.")
print("=" * 60)

# Prevent saving the same face every frame
last_face_save = 0
face_counter = 0

# Prevent writing thousands of observations
last_observation = 0


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    # -----------------------------------------------------
    # OBJECT DETECTION
    # -----------------------------------------------------

    results = model.predict(
        frame,
        imgsz=320,
        conf=0.40,
        verbose=False
    )

    objects_seen = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Object name
            object_name = model.names[class_id]

            objects_seen.append(
                f"{object_name} ({confidence * 100:.1f}%)"
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Robot label
            label = f"{object_name} {confidence * 100:.1f}%"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )


    # -----------------------------------------------------
    # FACE DETECTION
    # -----------------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Robot face counter
    face_count = len(faces)

    for i, (x, y, w, h) in enumerate(faces):

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "FACE",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # -------------------------------------------------
        # SAVE FACE IMAGE
        # -------------------------------------------------

        current_time = time.time()

        # Save at most once every 2 seconds
        if current_time - last_face_save >= 2:

            face_image = frame[
                y:y + h,
                x:x + w
            ]

            if face_image.size > 0:

                face_counter += 1

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                filename = (
                    f"face_{face_counter}_{timestamp}.jpg"
                )

                filepath = os.path.join(
                    face_folder,
                    filename
                )

                cv2.imwrite(
                    filepath,
                    face_image
                )

                print(
                    f"[ROBOT] Face captured -> {filepath}"
                )

                last_face_save = current_time


    # -----------------------------------------------------
    # ROBOT STATUS
    # -----------------------------------------------------

    status = (
        f"OBJECTS: {len(objects_seen)}   "
        f"FACES: {face_count}"
    )

    cv2.putText(
        frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # -----------------------------------------------------
    # ROBOT DESCRIPTION
    # -----------------------------------------------------

    current_time = time.time()

    if current_time - last_observation >= 2:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        if objects_seen:

            object_text = ", ".join(objects_seen)

            robot_message = (
                f"[{timestamp}] "
                f"ROBOT: I detect {object_text}. "
                f"I detect {face_count} face(s)."
            )

        else:

            robot_message = (
                f"[{timestamp}] "
                f"ROBOT: No recognized objects detected. "
                f"I detect {face_count} face(s)."
            )

        print(robot_message)

        with open(
            observation_file,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(robot_message + "\n")

        last_observation = current_time


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    cv2.imshow(.
        
        "ROBOT VISION",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# SHUTDOWN
# =========================================================

camera.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("ROBOT VISION SYSTEM OFFLINE")
print(f"Faces saved in: {face_folder}")
print(f"Observations saved in: {observation_file}")
print("=" * 60)