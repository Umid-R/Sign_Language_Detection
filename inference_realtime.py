import cv2
import numpy as np
import mediapipe as mp
import pickle
import collections
from tensorflow.keras.models import load_model
from config import (MODEL_PATH, LABEL_ENCODER,
                    CONFIDENCE_THRESHOLD,
                    SMOOTHING_BUFFER_SIZE,
                    FEATURE_SIZE)

# ── LOAD MODEL & ENCODER ──────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)

with open(LABEL_ENCODER, 'rb') as f:
    le = pickle.load(f)

print("Model loaded! ✅")
print(f"Classes: {le.classes_}")

# ── MEDIAPIPE SETUP ───────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
hands       = mp_hands.Hands(
    static_image_mode       = False,  # video mode
    max_num_hands           = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence  = 0.5
)

# ── NORMALIZATION ─────────────────────────────────────────────
def normalize_landmarks(landmarks):
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist  = points[0]
    points = points - wrist
    scale  = np.linalg.norm(points[9])
    if scale > 0:
        points = points / scale
    return points.flatten()

# ── SMOOTHING BUFFER ──────────────────────────────────────────
# Stores last N predictions and returns most common one
buffer = collections.deque(maxlen=SMOOTHING_BUFFER_SIZE)

def get_smoothed_prediction(prediction):
    buffer.append(prediction)
    # Return most common prediction in buffer
    most_common = collections.Counter(buffer).most_common(1)[0][0]
    return most_common

# ── DRAW OVERLAY ──────────────────────────────────────────────
def draw_overlay(frame, label, confidence, hand_detected):
    h, w = frame.shape[:2]

    if not hand_detected:
        # No hand message
        cv2.putText(frame,
            "No hand detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 255), 2)
        return frame

    if label == "Unknown":
        color = (0, 165, 255)   # orange for unknown
    else:
        color = (0, 255, 0)     # green for confident

    # ── Prediction label ──────────────────────────────────────
    cv2.putText(frame,
        f"Sign: {label}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0, color, 3)

    # ── Confidence percentage ─────────────────────────────────
    cv2.putText(frame,
        f"Confidence: {confidence*100:.1f}%",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, color, 2)

    # ── Confidence bar ────────────────────────────────────────
    bar_width = int(confidence * 300)
    cv2.rectangle(frame, (30, 125), (330, 150), (50, 50, 50), -1)
    cv2.rectangle(frame, (30, 125), (30 + bar_width, 150), color, -1)

    # ── Instructions ──────────────────────────────────────────
    cv2.putText(frame,
        "Press Q to quit",
        (30, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (200, 200, 200), 1)

    return frame

# ── SENTENCE BUILDER ──────────────────────────────────────────
sentence        = []
last_letter     = None
letter_counter  = 0
LETTER_HOLD     = 20   # frames to hold before adding to sentence

# ── MAIN LOOP ─────────────────────────────────────────────────
print("\nStarting webcam...")
print("Show your hand to the camera!")
print("Press Q to quit\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip like mirror
    frame     = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── RUN MEDIAPIPE ─────────────────────────────────────────
    results = hands.process(frame_rgb)

    hand_detected = False
    label         = "Unknown"
    confidence    = 0.0

    if results.multi_hand_landmarks:
        hand_detected = True
        landmarks     = results.multi_hand_landmarks[0]

        # Draw hand skeleton on frame
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # ── NORMALIZE & PREDICT ───────────────────────────────
        vector     = normalize_landmarks(landmarks.landmark)
        vector     = vector.reshape(1, FEATURE_SIZE)

        prediction = model.predict(vector, verbose=0)
        confidence = float(np.max(prediction))
        class_idx  = np.argmax(prediction)
        raw_label  = le.inverse_transform([class_idx])[0]

        # ── CONFIDENCE THRESHOLD ──────────────────────────────
        if confidence >= CONFIDENCE_THRESHOLD:
            label = get_smoothed_prediction(raw_label)
        else:
            label = "Unknown"
            buffer.clear()

        # ── SENTENCE BUILDER ──────────────────────────────────
        if label != "Unknown" and label != last_letter:
            letter_counter += 1
            if letter_counter >= LETTER_HOLD:
                sentence.append(label)
                last_letter    = label
                letter_counter = 0
        else:
            letter_counter = 0

    else:
        # Hand left frame — reset
        buffer.clear()
        last_letter    = None
        letter_counter = 0

    # ── DRAW OVERLAY ──────────────────────────────────────────
    frame = draw_overlay(frame, label, confidence, hand_detected)

    # ── SHOW SENTENCE ─────────────────────────────────────────
    sentence_str = " ".join(sentence[-10:])   # last 10 letters
    cv2.putText(frame,
        f"Sentence: {sentence_str}",
        (30, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 0), 2)

    # ── SHOW FRAME ────────────────────────────────────────────
    cv2.imshow("ASL Sign Language Detection", frame)

    # ── KEY CONTROLS ──────────────────────────────────────────
    key = cv2.waitKey(1)

    if key == ord('q'):          # quit
        break
    elif key == ord('c'):        # clear sentence
        sentence.clear()
        last_letter    = None
        letter_counter = 0
        print("Sentence cleared!")
    elif key == ord(' '):        # add space to sentence
        sentence.append(' ')

# ── CLEANUP ───────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("\nFinal sentence:", " ".join(sentence))
print("Bye bro! 🤙")