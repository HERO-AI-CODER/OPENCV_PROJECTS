import cv2
from ultralytics import YOLO
import time

# ---------------------------------------------------------
# LOAD AI MODEL
# ---------------------------------------------------------

print("Loading AI model...")

model = YOLO("yolo11n.pt")

print("AI model loaded.")


# ---------------------------------------------------------
# GET ITEM FROM USER
# ---------------------------------------------------------

required_item = input(
    "\nEnter the item you want to find: "
).strip().lower()

print("\nRequired item:", required_item)
print("Starting camera...")
print("Press Q to quit.")


# ---------------------------------------------------------
# OPEN CAMERA
# ---------------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()


found = False
last_found_time = 0


# ---------------------------------------------------------
# CAMERA LOOP
# ---------------------------------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read camera.")
        break


    # -----------------------------------------------------
    # RUN AI OBJECT DETECTION
    # -----------------------------------------------------

    results = model(frame, verbose=False)


    detected_required_item = False


    for result in results:

        boxes = result.boxes

        for box in boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            detected_name = model.names[
                class_id
            ].lower()


            # -------------------------------------------------
            # CHECK WHETHER DETECTED OBJECT IS REQUIRED ITEM
            # -------------------------------------------------

            if (
                detected_name == required_item
                and confidence > 0.45
            ):

                detected_required_item = True

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )


                label = (
                    f"{detected_name} "
                    f"{confidence * 100:.1f}%"
                )


                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


    # -----------------------------------------------------
    # DISPLAY STATUS
    # -----------------------------------------------------

    if detected_required_item:

        found = True
        last_found_time = time.time()

        cv2.putText(
            frame,
            "ITEM FOUND!",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    else:

        cv2.putText(
            frame,
            "Searching...",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )


    # -----------------------------------------------------
    # SHOW REQUIRED ITEM
    # -----------------------------------------------------

    cv2.putText(
        frame,
        "Required: " + required_item,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # -----------------------------------------------------
    # SHOW CAMERA
    # -----------------------------------------------------

    cv2.imshow(
        "Smart Shopping AI Camera",
        frame
    )


    # -----------------------------------------------------
    # QUIT
    # -----------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ---------------------------------------------------------
# CLOSE CAMERA
# ---------------------------------------------------------

camera.release()
cv2.destroyAllWindows()


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

if found:

    print("\n" + "=" * 50)
    print("ITEM IDENTIFIED")
    print("=" * 50)

    print("Required item:", required_item)
    print("Status: FOUND")

    print("\nNext step:")
    print("Searching for the best price...")

else:

    print("\nItem was not detected.")