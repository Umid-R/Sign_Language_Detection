import cv2
import os
import time
from config import GESTURES, RAW_DATA_PATH, SAMPLES_PER_CLASS, IMG_SIZE

def create_folders():
    """Create a folder for each gesture if not already there"""
    for gesture in GESTURES:
        folder = os.path.join(RAW_DATA_PATH, gesture)
        os.makedirs(folder, exist_ok=True)
    print("All folders ready!")

def collect_data():
    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam not found!")
        return

    print("Webcam opened successfully!")
    print("Press Q anytime to quit\n")

    # Loop through each gesture
    for gesture in GESTURES:
        save_path = os.path.join(RAW_DATA_PATH, gesture)

        # ── SKIP IF ALREADY COLLECTED ─────────────────────────
        existing = len(os.listdir(save_path))
        if existing >= SAMPLES_PER_CLASS:
            print(f"Skipping {gesture} — already has {existing} samples")
            continue

        # ── READY SCREEN ──────────────────────────────────────
        # Show a waiting screen before capturing starts
        print(f"\nGet ready to sign: {gesture}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame so it acts like a mirror
            frame = cv2.flip(frame, 1)

            # Show instruction on screen
            cv2.putText(frame,
                f"Get ready to sign: {gesture}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

            cv2.putText(frame,
                "Press SPACE when ready",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1)
            if key == ord(' '):   # space → start capturing
                break
            if key == ord('q'):   # q → quit everything
                cap.release()
                cv2.destroyAllWindows()
                return

        # ── COUNTDOWN ─────────────────────────────────────────
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            cv2.putText(frame,
                f"Starting in {countdown}...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 255), 3)

            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1000)   # wait 1 second per count

        # ── CAPTURE LOOP ──────────────────────────────────────
        count = 0
        existing = len(os.listdir(save_path))

        while count < SAMPLES_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Save the frame
            
            filename = os.path.join(save_path, f"frame_{existing + count:04d}.jpg")
            resized  = cv2.resize(frame, IMG_SIZE)
            cv2.imwrite(filename, resized)
            count += 1

            # Show progress on screen
            cv2.putText(frame,
                f"Signing: {gesture}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

            cv2.putText(frame,
                f"Captured: {count} / {SAMPLES_PER_CLASS}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

            # Progress bar
            bar_width = int((count / SAMPLES_PER_CLASS) * 400)
            cv2.rectangle(frame, (30, 130), (430, 160), (50, 50, 50), -1)
            cv2.rectangle(frame, (30, 130), (30 + bar_width, 160), (0, 255, 0), -1)

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        print(f"Done! {SAMPLES_PER_CLASS} frames saved for gesture: {gesture}")

    # ── ALL DONE ──────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("\nAll gestures collected successfully!")

# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    create_folders()
    collect_data()