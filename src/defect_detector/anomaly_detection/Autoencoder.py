# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 20:10:09 2025

@author: acer
"""

# autoencoder.py
# -------------------------------
# Convolutional Autoencoder
# For steel surface anomaly detection
# -------------------------------

from tensorflow.keras import layers, models


def build_autoencoder(input_shape=(200, 200, 1)):
    """
    Builds a simple convolutional autoencoder.

    Parameters:
        input_shape: tuple
            Shape of input image (H, W, C)

    Returns:
        autoencoder: keras Model
    """

    # -------------------
    # Encoder
    # -------------------
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    # Shape: (100, 100, 16)

    # Block 2
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    # Shape: (50, 50, 32)

    # Block 3
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)
    # Shape: (25, 25, 64)
    # This is the latent (compressed) representation

    # -------------------
    # Decoder
    # -------------------

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    # Shape: (50, 50, 64)

    # Block 2
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    # Shape: (100, 100, 32)

    # Block 3
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    # Shape: (200, 200, 16)

    # Output layer
    outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
    # Shape: (200, 200, 1)

    # -------------------
    # Model
    # -------------------
    autoencoder = models.Model(inputs, outputs, name="Steel_Autoencoder")

    return autoencoder
