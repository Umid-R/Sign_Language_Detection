#  Sign Language Detection System

A real-time American Sign Language (ASL) detection system that uses
hand pose estimation to recognize hand gestures and translate them
into letters and digits.

---

## 📋 Project Overview

This system uses MediaPipe Hands to extract skeletal hand landmarks
from webcam frames and feeds normalized pose vectors into a trained
neural network classifier. The final output runs as a real-time
inference pipeline on live webcam input.

### What It Recognizes
- All 26 ASL alphabet letters (A-Z)
- Digits 0-9
- Both left and right hands
- Real-time live webcam feed

### Key Results
- Overall test accuracy : 98.87%
- Average confidence    : 99.4%
- Real-time FPS         : 20+ FPS
- Model size            : < 10MB

---

## 📁 Project Structure
```
sign_language_detection/
│
├── data/
│   ├── raw/                    # Webcam captured frames
│   │   ├── A/                  # 3300 images per letter
│   │   ├── B/
│   │   │   ... (A-Z and 0-9)
│   │   └── 9/
│   └── poses/
│       ├── landmarks.csv       # Extracted pose vectors
│       ├── X_test.npy          # Test features
│       └── y_test.npy          # Test labels
│
├── models/
│   ├── best_model.h5           # Trained model
│   └── label_encoder.pkl       # Label encoder
│
├── plots/
│   ├── training_curves.png     # Loss and accuracy curves
│   └── confusion_matrix.png    # Per class confusion matrix
│
├── collect_data.py             # Data collection script
├── extract_poses.py            # Pose extraction script
├── train.py                    # Model training script
├── evaluate.py                 # Evaluation script
├── inference_realtime.py       # Real-time inference script
├── config.py                   # Configuration settings
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone or download the project
```bash
cd ~/Desktop
cd sign_language_detection
```

### 2. Create a fresh conda environment
```bash
conda create -n sign_lang python=3.10 -y
conda activate sign_lang
```

### 3. Install dependencies in correct order
```bash
pip install numpy==1.26.4
pip install protobuf==3.20.3
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.9
pip install tensorflow==2.13.0
pip install scikit-learn pandas matplotlib seaborn
```

### 4. Verify installation
```bash
python -c "import cv2; import mediapipe; import tensorflow; print('All good!')"
```

---

## 🚀 Usage Guide

### Always activate the environment first:
```bash
conda activate sign_lang
cd ~/Desktop/sign_language_detection
```

---

### Step 1 — Collect Your Own Data (Optional)
```bash
python collect_data.py
```
```
Controls:
    SPACEBAR → start capturing current gesture
    Q        → quit and resume later
    
Tips:
    - Print ASL chart and place next to laptop
    - Collect both right and left hand
    - Vary lighting and hand angle
    - Script automatically skips completed gestures
```

---

### Step 2 — Extract Pose Landmarks
```bash
python extract_poses.py
```
```
- Processes all images in data/raw/
- Extracts 21 hand landmarks per image
- Normalizes landmarks (wrist at origin)
- Saves 63-dimensional vectors to landmarks.csv
- Takes approximately 1-2 hours for full dataset
```

---

### Step 3 — Train The Model
```bash
python train.py
```
```
- Loads landmarks.csv
- Splits data 70/15/15 train/val/test
- Trains MLP neural network
- Saves best checkpoint to models/
- Plots training curves to plots/
- Takes approximately 5 minutes on M3 MacBook
```

---

### Step 4 — Evaluate The Model
```bash
python evaluate.py
```
```
- Loads best model
- Tests on held-out test set
- Prints per-class accuracy report
- Generates confusion matrix
- Flags weak gesture classes
```

---

### Step 5 — Run Real-Time Inference
```bash
python inference_realtime.py
```
```
Controls:
    Q        → quit
    C        → clear sentence buffer
    SPACEBAR → add space to sentence

Features:
    - Hand skeleton drawn on screen
    - Prediction label displayed
    - Confidence score and bar
    - Smoothing buffer (no flickering)
    - Sentence builder
```

---

## 🛠️ Configuration

All settings are in `config.py`:
```python
SAMPLES_PER_CLASS    = 300      # photos per gesture
BATCH_SIZE           = 32       # training batch size
EPOCHS               = 100      # max training epochs
LEARNING_RATE        = 0.001    # Adam optimizer lr
DROPOUT_RATE         = 0.3      # dropout regularization
PATIENCE             = 10       # early stopping patience
CONFIDENCE_THRESHOLD = 0.85     # min confidence for prediction
SMOOTHING_BUFFER_SIZE = 7       # frames for majority vote
```

---

## 📊 Dataset

| Source | Gestures | Images per class | Total |
|--------|----------|-----------------|-------|
| Kaggle ASL Alphabet | A-Z | 3000 | 78,000 |
| Custom webcam (right hand) | A-Z, 0-9 | 300 | 10,800 |
| Custom webcam (left hand) | A-Z, 0-9 | 300 | 10,800 |
| **Total** | **36 classes** | **~3300** | **88,800** |

### After Pose Extraction:
```
Total samples  : 72,901
Success rate   : 82.1%
Features       : 63 (21 landmarks × 3 coordinates)
```

---

## 🧠 Model Architecture
```
Input (63)
    ↓
Dense(256, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(64, ReLU)  → BatchNorm → Dropout(0.3)
    ↓
Dense(36, Softmax)
```
```
Total parameters : ~50,000
Model size       : < 5MB
Training time    : ~5 minutes on CPU
```

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Overall test accuracy | 98.87% |
| Average confidence | 99.4% |
| Letters A-Z accuracy | 94-100% |
| Digits 2-9 accuracy | 97-100% |
| Digit 0 accuracy | 88.2% |
| Digit 1 accuracy | 82.8% |

### Weak Classes
```
Digit 0 → 88.2% (slightly below 90% target)
Digit 1 → 82.8% (limited training samples)

Fix: collect more data for digits 0 and 1
```

---

## 🔧 Tech Stack

| Component | Tool |
|-----------|------|
| Pose Extraction | MediaPipe Hands |
| Video Capture | OpenCV |
| Data Handling | NumPy, Pandas |
| Model | TensorFlow/Keras MLP |
| Visualization | Matplotlib, Seaborn |
| Environment | Conda (sign_lang) |

---

## ⚠️ Known Issues & Fixes

### MediaPipe/OpenCV conflict on Mac
```bash
# Use the sign_lang conda environment
conda activate sign_lang
```

### NumPy version conflict
```bash
pip install numpy==1.26.4
```

### Model loading error across TF versions
```bash
# Always train and run in same environment
conda activate sign_lang
python train.py        # retrain if needed
python inference_realtime.py
```

---

## 📝 Notes
- J and Z are motion-based gestures and may be less accurate
  as they require movement rather than static poses
- Digit 1 has fewer training samples — collect more for better accuracy
- System works best in good lighting with hand clearly visible
- Both left and right hands are supported

---

## 👤 Author
Built as part of a Sign Language Detection assignment
using MediaPipe, TensorFlow, and OpenCV
```

---

Save it bro! That's a really solid README 🤙

---

## Now You're Fully Done! ✅
```
✅ Phase 1 — Setup
✅ Phase 2 — Data collection
✅ Phase 3 — Pose extraction
✅ Phase 4 — Model training (98.87%!)
✅ Phase 5 — Evaluation
✅ Phase 6 — Real-time inference
✅ Phase 7 — README done!
