import os
import cv2

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

LETTERS = ["N", "S", "Z"]
dataset_size = 150

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to (1) or (2).")

for letter in LETTERS:
    class_dir = os.path.join(DATA_DIR, letter)
    os.makedirs(class_dir, exist_ok=True)

    print(f"Collecting data for class {letter}")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("frame", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        cv2.imshow("frame", frame)
        cv2.waitKey(30)

        cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
