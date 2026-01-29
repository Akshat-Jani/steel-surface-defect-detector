# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 13:13:22 2025

@author: acer
"""

# infer_anomaly.py
# ------------------------------------
# Inference + anomaly heatmap generation
# ------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# For running locally
# =============================================================================
# # ---------------- CONFIG ----------------
# MODEL_PATH = "steel_autoencoder.keras"
# TEST_IMAGE_PATH = r"D:\Python_Programs\Projects\Steel-surface-defect-detector\data\validation_data\inclusion\inclusion_299.jpg"
# IMG_SIZE = (200, 200)
# # --------------------------------------
# 
# 
# def load_and_preprocess_image(path):
#     """Load image, convert to grayscale, resize, normalize"""
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError("Could not read image")
# 
#     img = cv2.resize(img, IMG_SIZE)
#     img = img.astype("float32") / 255.0
#     img = np.expand_dims(img, axis=-1)  # (H, W, 1)
#     img = np.expand_dims(img, axis=0)   # (1, H, W, 1)
# 
#     return img
# 
# 
# def compute_anomaly_map(original, reconstructed):
#     """Absolute reconstruction error"""
#     return np.abs(original - reconstructed)
# 
# 
# # ---------------- LOAD MODEL ----------------
# print("[INFO] Loading autoencoder...")
# autoencoder = load_model(MODEL_PATH)
# 
# 
# # ---------------- LOAD IMAGE ----------------
# print("[INFO] Loading test image...")
# x = load_and_preprocess_image(TEST_IMAGE_PATH)
# 
# 
# # ---------------- RECONSTRUCTION ----------------
# print("[INFO] Reconstructing image...")
# x_recon = autoencoder.predict(x)
# 
# 
# # ---------------- ANOMALY MAP ----------------
# anomaly_map = compute_anomaly_map(x, x_recon)
# 
# 
# # ---------------- VISUALIZATION ----------------
# orig = x[0, :, :, 0]
# recon = x_recon[0, :, :, 0]
# anomaly = anomaly_map[0, :, :, 0]
# 
# plt.figure(figsize=(12, 4))
# 
# plt.subplot(1, 3, 1)
# plt.title("Original")
# plt.imshow(orig, cmap="gray")
# plt.axis("off")
# 
# plt.subplot(1, 3, 2)
# plt.title("Reconstruction")
# plt.imshow(recon, cmap="gray")
# plt.axis("off")
# 
# plt.subplot(1, 3, 3)
# plt.title("Anomaly Map")
# plt.imshow(anomaly, cmap="hot")
# plt.axis("off")
# 
# plt.tight_layout()
# plt.show()
# =============================================================================

def anomaly_score_and_map(img_gray_200, autoencoder):
    x = np.expand_dims(img_gray_200, axis=(0, -1))  # (1, 200, 200, 1)

    recon = autoencoder.predict(x, verbose=0)

    anomaly_map = np.abs(x - recon)[0, :, :, 0]
    score = anomaly_map.mean()

    reconstructed = recon[0, :, :, 0]  # (200, 200)

    return score, anomaly_map, reconstructed
