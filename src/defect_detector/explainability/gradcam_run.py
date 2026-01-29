# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 20:05:13 2025

@author: acer
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from src.defect_detector.explainability.gradcam import gradcam_on_image

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"D:\Python_Programs\Projects\Steel-surface-defect-detector\models\mobilenetv2_defect.keras"
IMAGE_PATH = r"D:\Python_Programs\Projects\Steel-surface-defect-detector\data\raw\scratches\scratches_52.jpg"
OUTPUT_PATH = "gradcam_output.png"
IMG_SIZE = (224, 224)

# -----------------------------
# Load model
# -----------------------------
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded")

# -----------------------------
# Load image
# -----------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize(IMG_SIZE)
img_array = np.array(img)

# -----------------------------
# Run Grad-CAM
# -----------------------------
print("[INFO] Running Grad-CAM...")
overlay = gradcam_on_image(
    model=model,
    img_array=img_array,
    preprocess_fn=tf.keras.applications.mobilenet_v2.preprocess_input,
    target_size=IMG_SIZE,
    alpha=0.4
)

# -----------------------------
# Save output
# -----------------------------
cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"[SUCCESS] Grad-CAM saved as {OUTPUT_PATH}")
