# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 22:14:59 2025

@author: acer
"""

"""
explainability/gradcam.py
A simple, robust Grad-CAM utility that works with:
 - flat Keras models (conv layers at top-level)
 - nested models (backbone saved as Functional inside the top model)
 - single-input and multi-input models

Public API:
 - gradcam_on_image(model, img_array, target_layer=None, preprocess_fn=None, target_size=(224,224), alpha=0.4)

Returns:
 - overlay_rgb: uint8 HxWx3 image (RGB) with heatmap blended
 - heatmap_resized: float HxW heatmap normalized to [0,1]

Keep this file readable: lots of inline comments for teaching.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from PIL import Image


def gradcam_on_image(
    model,
    img_array,
    preprocess_fn,
    target_size=(224, 224),
    alpha=0.4
):
    """
    Simple Grad-CAM for MobileNetV2-based models (including nested backbone).

    Inputs:
    - model: trained Keras model (.keras)
    - img_array: HxWx3 RGB image (numpy)
    - preprocess_fn: preprocessing used during training
    - target_size: input size of model
    - alpha: overlay strength

    Returns:
    - overlay image (uint8 RGB)
    """

    # ---------------------------
    # 1. Prepare image
    # ---------------------------
    img = Image.fromarray(img_array.astype(np.uint8))
    img = img.resize(target_size)
    img = np.array(img).astype(np.float32)

    original = img.copy()
    img = preprocess_fn(img)
    img = np.expand_dims(img, axis=0)

    # ---------------------------
    # 2. Extract MobileNet backbone
    # ---------------------------
    backbone = None
    for layer in model.layers:
        if layer.__class__.__name__ in ("Functional", "Model") and "mobilenet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise ValueError("MobileNet backbone not found")

    # ---------------------------
    # 3. Choose last convolution layer
    # ---------------------------
    conv_layer_name = "Conv_1"
    conv_layer = backbone.get_layer(conv_layer_name)

    # ---------------------------
    # 4. Rebuild model: input → conv output → final prediction
    # ---------------------------
    x = backbone.output
    backbone_index = model.layers.index(backbone)

    for layer in model.layers[backbone_index + 1:]:
        x = layer(x)

    grad_model = Model(
        inputs=backbone.input,
        outputs=[conv_layer.output, x]
    )

    # ---------------------------
    # 5. Compute Grad-CAM
    # ---------------------------
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy()
    heatmap /= (np.max(heatmap) + 1e-8)
    
    # ---------------------------
    # 6. Overlay heatmap
    # ---------------------------
        

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]),interpolation=cv2.INTER_CUBIC)
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original.astype(np.uint8), 1 - alpha, heatmap, alpha, 0)

    return overlay
