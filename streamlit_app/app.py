# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:06:13 2025

@author: acer
"""

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ---------------- LOAD MODELS ----------------
autoencoder = load_model(r"D:\Python_Programs\Projects\Steel-surface-defect-detector\models\steel_autoencoder.keras")
classifier_model = load_model(r"D:\Python_Programs\Projects\Steel-surface-defect-detector\models\mobilenetv2_defect.keras")

from defect_detector.explainability.gradcam import gradcam_on_image
from defect_detector.anomaly_detection.infer_anomaly import anomaly_score_and_map
from defect_detector.defect_classification.cnn.inference import predict_defect

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ---------------- CONFIG ----------------
ANOMALY_THRESHOLD = 0.013   # fine tuning (screen_shot for reference in anomaly_detection folder)
# --------------------------------------


st.title("Industrial Steel Surface Inspection")

uploaded_file = st.file_uploader("Upload steel surface image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Input Image")
    st.image(img_rgb, use_column_width=True)

    # ---------------- ANOMALY DETECTION ----------------
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.resize(img_gray, (200, 200))
    img_gray = img_gray.astype("float32") / 255.0

    score, anomaly_map, reconstructed = anomaly_score_and_map(img_gray, autoencoder)

    st.subheader("Anomaly Check")
    st.write(f"Anomaly score: **{score:.4f}**")
    
    st.subheader("Autoencoder Reconstruction")
    st.image(reconstructed, caption="Reconstructed Normal Surface", clamp=True)

    st.subheader("Anomaly Map")
    st.image(anomaly_map, caption="Reconstruction Error", clamp=True)

    if score < ANOMALY_THRESHOLD:
        st.success("✅ Normal steel surface detected. No defect.")
    else:
        st.error("⚠ Defect detected. Running classifier...")

        # ---------------- CLASSIFICATION ----------------
        label, confidence = predict_defect(img_bgr)

        st.subheader("Defect Classification")
        st.write(f"**Defect type:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # ---------------- GRAD-CAM ----------------
        st.subheader("Grad-CAM Explanation")

        overlay = gradcam_on_image(
            classifier_model,
            img_rgb,
            preprocess_fn=preprocess_input,
            target_size=(224, 224)
        )

        st.image(overlay, caption="Grad-CAM Visualization", use_column_width=True)
