import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from config import (LANDMARKS_CSV, MODEL_PATH, LABEL_ENCODER,
                    PLOTS_PATH, BATCH_SIZE, EPOCHS,
                    LEARNING_RATE, DROPOUT_RATE, PATIENCE)

# ── REPRODUCIBILITY ───────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── STEP 1: LOAD DATA ─────────────────────────────────────────
print("="*50)
print("STEP 1: Loading data...")
print("="*50)

df = pd.read_csv(LANDMARKS_CSV)
print(f"Total samples loaded : {len(df)}")
print(f"Total columns        : {len(df.columns)}")
print(f"\nSamples per class:")
print(df['label'].value_counts().sort_index())

# Split features and labels
X = df.drop('label', axis=1).values   # 63 numbers
y = df['label'].values                 # A, B, C...

# ── STEP 2: ENCODE LABELS ─────────────────────────────────────
print("\n" + "="*50)
print("STEP 2: Encoding labels...")
print("="*50)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

print(f"Number of classes : {num_classes}")
print(f"Classes           : {le.classes_}")

# Save label encoder
os.makedirs(os.path.dirname(LABEL_ENCODER), exist_ok=True)
with open(LABEL_ENCODER, 'wb') as f:
    pickle.dump(le, f)
print(f"Label encoder saved to: {LABEL_ENCODER}")

# One hot encode for keras
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# ── STEP 3: SPLIT DATA ────────────────────────────────────────
print("\n" + "="*50)
print("STEP 3: Splitting data...")
print("="*50)

# First split: 85% train+val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_categorical,
    test_size    = 0.15,
    random_state = 42,
    stratify     = y_encoded
)

# Second split: 70% train, 15% val
y_temp_labels = le.inverse_transform(np.argmax(y_temp, axis=1))
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size    = 0.176,   # 0.176 of 85% ≈ 15% of total
    random_state = 42,
    stratify     = y_temp_labels
)

print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Test samples       : {len(X_test)}")

# Save test set for evaluate.py
np.save('data/poses/X_test.npy', X_test)
np.save('data/poses/y_test.npy', y_test)
print("Test set saved for evaluate.py")

# ── STEP 4: CLASS WEIGHTS ─────────────────────────────────────
print("\n" + "="*50)
print("STEP 4: Computing class weights...")
print("="*50)

y_train_labels = np.argmax(y_train, axis=1)
class_weights  = compute_class_weight(
    class_weight = 'balanced',
    classes      = np.unique(y_train_labels),
    y            = y_train_labels
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights computed ✅")

# ── STEP 5: BUILD MODEL ───────────────────────────────────────
print("\n" + "="*50)
print("STEP 5: Building model...")
print("="*50)

model = Sequential([
    # Input layer
    Dense(256, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(DROPOUT_RATE),

    # Hidden layer 1
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(DROPOUT_RATE),

    # Hidden layer 2
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(DROPOUT_RATE),

    # Output layer
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()

# ── STEP 6: CALLBACKS ─────────────────────────────────────────
print("\n" + "="*50)
print("STEP 6: Setting up callbacks...")
print("="*50)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

callbacks = [
    # Save best model
    ModelCheckpoint(
        MODEL_PATH,
        monitor           = 'val_accuracy',
        save_best_only    = True,
        verbose           = 1
    ),
    # Stop early if no improvement
    EarlyStopping(
        monitor           = 'val_accuracy',
        patience          = PATIENCE,
        restore_best_weights = True,
        verbose           = 1
    ),
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor           = 'val_loss',
        factor            = 0.5,
        patience          = 5,
        min_lr            = 1e-6,
        verbose           = 1
    )
]

print("Callbacks ready ✅")

# ── STEP 7: TRAIN ─────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 7: Training model...")
print("="*50)

history = model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    class_weight    = class_weight_dict,
    callbacks       = callbacks,
    verbose         = 1
)

# ── STEP 8: PLOT TRAINING CURVES ──────────────────────────────
print("\n" + "="*50)
print("STEP 8: Plotting training curves...")
print("="*50)

os.makedirs(PLOTS_PATH, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'],     label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'training_curves.png'))
print(f"Training curves saved to {PLOTS_PATH}/training_curves.png")

# ── FINAL REPORT ──────────────────────────────────────────────
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)

final_train_acc = history.history['accuracy'][-1]
final_val_acc   = history.history['val_accuracy'][-1]

print(f"Final train accuracy : {final_train_acc*100:.2f}%")
print(f"Final val accuracy   : {final_val_acc*100:.2f}%")
print(f"Model saved to       : {MODEL_PATH}")
print(f"Total epochs ran     : {len(history.history['accuracy'])}")