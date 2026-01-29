# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 12:02:51 2025

@author: acer
"""

# threshold_test.py
# ----------------------------------------
# Manual threshold tuning for anomaly score
# ----------------------------------------

import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from infer_anomaly import anomaly_score_and_map


# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\Python_Programs\Projects\Steel-surface-defect-detector\anomaly_detection\steel_autoencoder.keras"
TEST_FOLDER = r"D:\New folder"  # change this
IMG_SIZE = (200, 200)
# ---------------------------------------


def load_gray_200(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return img


# ---------------- LOAD MODEL ----------------
print("[INFO] Loading autoencoder...")
autoencoder = load_model(MODEL_PATH)


# ---------------- LOOP THROUGH IMAGES ----------------
image_files = [
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print(f"[INFO] Found {len(image_files)} images")

for fname in image_files:
    img_path = os.path.join(TEST_FOLDER, fname)

    img_gray = load_gray_200(img_path)
    if img_gray is None:
        continue

    score, anomaly_map, recon = anomaly_score_and_map(img_gray, autoencoder)

    print(f"\nImage: {fname}")
    print(f"Anomaly score: {score:.5f}")

    # ---------------- VISUALIZATION ----------------
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img_gray, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Reconstruction")
    plt.imshow(recon, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Anomaly Map")
    plt.imshow(anomaly_map, cmap="hot")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
