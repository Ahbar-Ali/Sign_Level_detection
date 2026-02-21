import pickle
from collections import deque, Counter
import cv2
import mediapipe as mp
import numpy as np

CAM_INDEX = 1  
USE_AVFOUNDATION = True  

CONF_THRESHOLD = 0.70     
SMOOTH_WINDOW = 10       
STABLE_FRAMES = 8         

model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]
classes = model_dict.get("labels", None)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,        
    max_num_hands=1,                
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def extract_features_and_bbox(hand_landmarks, W, H):
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)

    xs = pts[:, 0]
    ys = pts[:, 1]
    x1 = int(xs.min() * W) - 10
    y1 = int(ys.min() * H) - 10
    x2 = int(xs.max() * W) + 10
    y2 = int(ys.max() * H) + 10

    wrist = pts[0].copy()
    pts = pts - wrist
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale

    feats = pts.flatten()

    return feats, (x1, y1, x2, y2)

def majority_vote(items):
    return Counter(items).most_common(1)[0][0]


if USE_AVFOUNDATION:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    raise RuntimeError(f"Could not open webcam at index {CAM_INDEX}. Try 0/1/2.")


# Word-building state
pred_queue = deque(maxlen=SMOOTH_WINDOW)

stable_label = None
stable_count = 0

current_word = ""

last_accepted_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display_label = "No hand"
    display_conf = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # draw landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        feats, (x1, y1, x2, y2) = extract_features_and_bbox(hand_landmarks, W, H)

        if feats.shape[0] == 42:
            proba = model.predict_proba([feats])[0]
            best_idx = int(np.argmax(proba))
            best_conf = float(proba[best_idx])

            raw_label = classes[best_idx] if classes is not None else str(best_idx)

            if best_conf < CONF_THRESHOLD:
                label_now = "Unknown"
            else:
                label_now = raw_label

            display_label = label_now
            display_conf = best_conf

            pred_queue.append(label_now)
            smooth_label = majority_vote(pred_queue)

            if smooth_label == stable_label:
                stable_count += 1
            else:
                stable_label = smooth_label
                stable_count = 1
            
            if smooth_label in ["Unknown", "No hand"]:
                last_accepted_label = None

            
            if stable_label not in ["Unknown", "No hand"] and stable_count >= STABLE_FRAMES:
                if stable_label != last_accepted_label:
                    if stable_label == "SPACE":
                        if current_word and not current_word.endswith(" "):
                            current_word += " "
                    else:
                        current_word += stable_label
                    last_accepted_label = stable_label
            
           
            cv2.putText(frame, f"{smooth_label} ({best_conf:.2f})", (x1, max(30, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)


    cv2.rectangle(frame, (0, 0), (W, 110), (40, 40, 40), -1)
    cv2.putText(frame, "Keys: [b]=backspace  [c]=clear  [q]=quit" , (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, current_word, (20,95),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow("inference", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        current_word = ""
        last_accepted_label = None
    elif key == ord("b"):
        current_word = current_word[:-1]
        last_accepted_label = None

cap.release()
cv2.destroyAllWindows()