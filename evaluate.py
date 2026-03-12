import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
from config import (MODEL_PATH, LABEL_ENCODER,
                    PLOTS_PATH, CONFIDENCE_THRESHOLD)

# ── STEP 1: LOAD MODEL & ENCODER ─────────────────────────────
print("="*50)
print("STEP 1: Loading model and encoder...")
print("="*50)

model = load_model(MODEL_PATH)

with open(LABEL_ENCODER, 'rb') as f:
    le = pickle.load(f)

print(f"Model loaded from    : {MODEL_PATH}")
print(f"Classes              : {le.classes_}")

# ── STEP 2: LOAD TEST DATA ────────────────────────────────────
print("\n" + "="*50)
print("STEP 2: Loading test data...")
print("="*50)

X_test = np.load('data/poses/X_test.npy')
y_test = np.load('data/poses/y_test.npy')

print(f"Test samples         : {len(X_test)}")

# ── STEP 3: MAKE PREDICTIONS ──────────────────────────────────
print("\n" + "="*50)
print("STEP 3: Making predictions...")
print("="*50)

# Get prediction probabilities
y_pred_probs = model.predict(X_test)

# Get predicted class index
y_pred_index = np.argmax(y_pred_probs, axis=1)
y_true_index = np.argmax(y_test,       axis=1)

# Convert back to labels
y_pred_labels = le.inverse_transform(y_pred_index)
y_true_labels = le.inverse_transform(y_true_index)

# Get confidence scores
y_confidence  = np.max(y_pred_probs, axis=1)

# ── STEP 4: OVERALL ACCURACY ──────────────────────────────────
print("\n" + "="*50)
print("STEP 4: Overall accuracy...")
print("="*50)

overall_acc = accuracy_score(y_true_index, y_pred_index)
print(f"Overall Test Accuracy : {overall_acc*100:.2f}%")

if overall_acc >= 0.95:
    print("✅ PASSED — meets 95% requirement!")
else:
    print("❌ FAILED — below 95% requirement")

# ── STEP 5: PER CLASS REPORT ──────────────────────────────────
print("\n" + "="*50)
print("STEP 5: Per class report...")
print("="*50)

report = classification_report(
    y_true_labels,
    y_pred_labels,
    target_names=le.classes_
)
print(report)

# ── STEP 6: FLAG WEAK CLASSES ─────────────────────────────────
print("\n" + "="*50)
print("STEP 6: Flagging weak classes...")
print("="*50)

report_dict = classification_report(
    y_true_labels,
    y_pred_labels,
    target_names = le.classes_,
    output_dict  = True
)

print("\nClasses below 90% accuracy:")
weak_classes = []
for gesture in le.classes_:
    acc = report_dict[gesture]['precision']
    if acc < 0.90:
        weak_classes.append(gesture)
        print(f"  ❌ {gesture} : {acc*100:.1f}%")

if not weak_classes:
    print("  ✅ All classes above 90%!")

# ── STEP 7: CONFUSION MATRIX ──────────────────────────────────
print("\n" + "="*50)
print("STEP 7: Generating confusion matrix...")
print("="*50)

cm = confusion_matrix(y_true_labels, y_pred_labels,
                      labels=le.classes_)

plt.figure(figsize=(20, 16))
sns.heatmap(
    cm,
    annot      = True,
    fmt        = 'd',
    cmap       = 'Blues',
    xticklabels = le.classes_,
    yticklabels = le.classes_
)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('True Label',      fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

os.makedirs(PLOTS_PATH, exist_ok=True)
cm_path = os.path.join(PLOTS_PATH, 'confusion_matrix.png')
plt.savefig(cm_path)
print(f"Confusion matrix saved to: {cm_path}")

# ── STEP 8: CONFIDENCE ANALYSIS ───────────────────────────────
print("\n" + "="*50)
print("STEP 8: Confidence analysis...")
print("="*50)

avg_confidence = np.mean(y_confidence)
low_confidence = np.sum(y_confidence < CONFIDENCE_THRESHOLD)

print(f"Average confidence    : {avg_confidence*100:.1f}%")
print(f"Low confidence preds  : {low_confidence}")
print(f"Confidence threshold  : {CONFIDENCE_THRESHOLD*100:.0f}%")

# ── FINAL REPORT ──────────────────────────────────────────────
print("\n" + "="*50)
print("EVALUATION COMPLETE")
print("="*50)
print(f"Overall accuracy      : {overall_acc*100:.2f}%")
print(f"Weak classes          : {weak_classes if weak_classes else 'None!'}")
print(f"Confusion matrix      : {cm_path}")
print(f"Average confidence    : {avg_confidence*100:.1f}%")