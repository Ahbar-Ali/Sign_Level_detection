import os
import pickle
import cv2
import mediapipe as mp
import numpy as np


print("mediapipe file:", mp.__file__)
print("mediapipe version:", getattr(mp, "__version__", "no version attr"))
print("has solutions?", hasattr(mp, "solutions"))
print("dir contains solutions?", "solutions" in dir(mp))

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = "./data"

data = []
labels = []

def extract_features(hand_landmarks) -> list:
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist

    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale

    return pts.flatten().tolist()



for dir_name in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, dir_name)
    
    if not os.path.isdir(class_dir):
        continue

    for img_name in sorted(os.listdir(class_dir)):
        img_path = os.path.join(class_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        feats = extract_features(hand_landmarks)

        if len(feats) == 42:
            data.append(feats)
            labels.append(dir_name)

hands.close()

with open("data.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print(f"Saved {len(data)} samples to data.pickle")
