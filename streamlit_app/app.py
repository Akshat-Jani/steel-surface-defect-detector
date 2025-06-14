# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:02:31 2025

@author: acer
"""
'''
Streamlit is an open-source Python library for building simple, interactive web apps — especially for
 ML, data science, and AI projects.
'''
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load model once
model = load_model("models/mobilenetv2_defect.keras")
class_indices = {0: "crazing", 1: "inclusion", 2: "patches", 3: "pitted_surface", 4: "rolled-in_scale", 5: "scratches"}

# Prediction function
def predict_image(img):
    img = img.convert("RGB") 
    '''
    When you upload a .png or .jpeg image with transparency (common on the web), PIL.Image.open() 
    gives it a 4-channel RGBA image instead of the expected 3-channel RGB.
    convert("RGB") removes the 4th (alpha) channel and ensures compatibility with MobileNet
    '''
    
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    pred_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_indices[pred_index], confidence

# Streamlit UI
st.title("Steel Surface Defect Detector") 
#Adds a big header at the top of the web app


uploaded_file = st.file_uploader("Upload a defect image", type=["jpg", "png", "jpeg"])
# Adds a drag-and-drop or click-to-upload file input
#Only allows image formats

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    label, conf = predict_image(img)
    st.markdown(f"### Predicted Class: `{label}`")
    st.markdown(f"**Confidence:** {conf*100:.2f}%")

'''
Run this on prompt after running this : streamlit run streamlit_app/app.py
'''