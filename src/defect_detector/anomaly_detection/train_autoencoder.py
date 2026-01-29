# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 11:15:41 2025

@author: acer
"""

# train_autoencoder.py
# ------------------------------------
# Train convolutional autoencoder
# on normal steel surface images
# ------------------------------------

import os
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from Autoencoder import build_autoencoder


# ---------------- CONFIG ----------------
DATA_DIR = "D:\Python_Programs\Projects\Steel-surface-defect-detector\data\Clean_Steel_Surface\processed"    
IMG_SIZE = (200, 200)
BATCH_SIZE = 8
EPOCHS = 30
MODEL_OUT = r"D:\Python_Programs\Projects\Steel-surface-defect-detector\models\steel_autoencoder.keras"
# --------------------------------------


def load_images(folder):
    images = []

    for fname in os.listdir(folder):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = img.astype("float32") / 255.0  # normalize to [0,1]
        img = np.expand_dims(img, axis=-1)   # (H, W, 1)
        images.append(img)

    return np.array(images)


# ---------------- LOAD DATA ----------------
print("[INFO] Loading normal steel images...")
x_train = load_images(DATA_DIR)
print(f"[INFO] Loaded {len(x_train)} images")
print(f"[INFO] Image shape: {x_train.shape}")


# ---------------- BUILD MODEL ----------------
autoencoder = build_autoencoder(input_shape=(200, 200, 1))
autoencoder.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse"
)

autoencoder.summary()


# ---------------- TRAIN ----------------
checkpoint = ModelCheckpoint(
    MODEL_OUT,
    monitor="loss",
    save_best_only=True,
    verbose=1
)

print("[INFO] Starting training...")
autoencoder.fit(
    x_train,
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[checkpoint]
)

print(f"[INFO] Training complete. Model saved as {MODEL_OUT}")
