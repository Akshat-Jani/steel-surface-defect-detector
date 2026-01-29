# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 18:35:25 2025

@author: acer
"""

# classifier/inference.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model ONCE
model = load_model(r"D:\Python_Programs\Projects\Steel-surface-defect-detector\models\mobilenetv2_defect.keras")

class_indices = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches"
}


def predict_defect(img_bgr):
    """
    img_bgr: OpenCV image (H, W, 3)
    """

    img = cv2.resize(img_bgr, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)

    class_id = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    return class_indices[class_id], confidence
