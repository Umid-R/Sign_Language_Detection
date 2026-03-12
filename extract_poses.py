import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from config import GESTURES, RAW_DATA_PATH, LANDMARKS_CSV

# ── MEDIAPIPE SETUP ───────────────────────────────────────────
mp_hands   = mp.solutions.hands
hands      = mp_hands.Hands(
    static_image_mode    = True,   # we're processing photos not video
    max_num_hands        = 1,      # only detect one hand per image
    min_detection_confidence = 0.3 # low threshold to catch more hands
)

# ── NORMALIZATION FUNCTION ────────────────────────────────────
def normalize_landmarks(landmarks):
    """
    Normalize 21 landmarks:
    1. Move wrist to origin (0,0,0)
    2. Scale by wrist to middle finger MCP distance
    3. Flatten to 63 numbers
    """
    # Convert to numpy array
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # Step 1 — subtract wrist position (landmark 0)
    wrist  = points[0]
    points = points - wrist

    # Step 2 — scale by wrist to middle finger MCP distance
    # middle finger MCP = landmark 9
    scale  = np.linalg.norm(points[9])
    if scale > 0:
        points = points / scale

    # Step 3 — flatten to 63 numbers
    return points.flatten()

# ── MAIN EXTRACTION FUNCTION ──────────────────────────────────
def extract_poses():
    data      = []   # will hold all rows
    total     = 0    # total images processed
    success   = 0    # successfully extracted
    failed    = 0    # failed (no hand detected)

    print("Starting pose extraction...")
    print(f"Processing {len(GESTURES)} gesture classes\n")

    for gesture in GESTURES:
        folder       = os.path.join(RAW_DATA_PATH, gesture)
        images       = os.listdir(folder)
        class_success = 0
        class_failed  = 0

        print(f"Processing gesture: {gesture} ({len(images)} images)")

        for img_file in images:
            img_path = os.path.join(folder, img_file)

            # Read image
            image = cv2.imread(img_path)
            if image is None:
                class_failed += 1
                continue

            # Convert BGR to RGB (MediaPipe needs RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run MediaPipe
            results = hands.process(image_rgb)

            # Check if hand was detected
            if results.multi_hand_landmarks:
                # Get first hand landmarks
                landmarks = results.multi_hand_landmarks[0].landmark

                # Normalize landmarks
                vector = normalize_landmarks(landmarks)

                # Append vector + class label
                row = list(vector) + [gesture]
                data.append(row)
                class_success += 1
            else:
                class_failed += 1

        total   += len(images)
        success += class_success
        failed  += class_failed

        # Print per class stats
        rate = (class_success / len(images)) * 100 if len(images) > 0 else 0
        print(f"  ✅ Success: {class_success}  ❌ Failed: {class_failed}  Rate: {rate:.1f}%")

    # ── SAVE TO CSV ───────────────────────────────────────────
    print(f"\nSaving to {LANDMARKS_CSV}...")

    # Build column names
    columns = []
    for i in range(21):
        columns += [f"x{i}", f"y{i}", f"z{i}"]
    columns.append("label")

    # Create dataframe and save
    df = pd.DataFrame(data, columns=columns)
    os.makedirs(os.path.dirname(LANDMARKS_CSV), exist_ok=True)
    df.to_csv(LANDMARKS_CSV, index=False)

    # ── FINAL REPORT ──────────────────────────────────────────
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total images processed : {total}")
    print(f"Successfully extracted : {success}")
    print(f"Failed (no hand)       : {failed}")
    print(f"Overall success rate   : {(success/total)*100:.1f}%")
    print(f"\nDataset saved to       : {LANDMARKS_CSV}")
    print(f"Total rows in CSV      : {len(df)}")
    print(f"Columns                : {len(df.columns)}")
    print("\nSamples per class:")
    print(df['label'].value_counts().sort_index())

# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    extract_poses()