# ─── Gesture Labels ───────────────────────────────────────
GESTURES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# ─── Data Paths ───────────────────────────────────────────
RAW_DATA_PATH     = 'data/raw'
POSES_PATH        = 'data/poses'
LANDMARKS_CSV     = 'data/poses/landmarks.csv'

# ─── Model Paths ──────────────────────────────────────────
MODEL_PATH        = 'models/best_model.h5'
LABEL_ENCODER     = 'models/label_encoder.pkl'

# ─── Plots Path ───────────────────────────────────────────
PLOTS_PATH        = 'plots'

# ─── Data Collection ──────────────────────────────────────
SAMPLES_PER_CLASS = 300          # frames to capture per gesture
IMG_SIZE          = (200, 200)   # saved frame size

# ─── Model Hyperparameters ────────────────────────────────
BATCH_SIZE        = 32
EPOCHS            = 100
LEARNING_RATE     = 0.001
DROPOUT_RATE      = 0.3
PATIENCE          = 10           # early stopping patience

# ─── Inference ────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.85    # below this → "Unknown"
SMOOTHING_BUFFER_SIZE = 7       # frames for majority vote
NUM_LANDMARKS         = 21
LANDMARK_DIMS         = 3
FEATURE_SIZE          = 63      # 21 × 3