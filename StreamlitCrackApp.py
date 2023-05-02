# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:23:03 2023

@author: osjostedt
"""

import requests
import ModelLoading
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import io
from keras.models import load_model
import tensorflow.keras.backend as K

"""
Run streamlit app by running: streamlit run StreamlitCrackApp.py
"""

models_path = 'Finished Models'
model_filename = 'Unet_500epochs_2.h5'
histogram_equalization_key = False
continue_learning = False
generate_images_key = False

def f1_score(y_true, y_pred):
    y_true = K.flatten(K.round(y_true))
    y_pred = K.flatten(K.round(y_pred))
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1_score

new_model = load_model(f'{models_path}/{model_filename}', custom_objects={'f1_score': f1_score}, compile=False)

# Set the URL of your Flask API endpoint
# API_URL = 'http://localhost:5000/crackpred'

# Set the title and description for your Streamlit app
st.title('Crack Detection App')
st.write('Upload an image to detect cracks')

# Add an image upload field to the Streamlit app
uploaded_file = st.file_uploader('Choose an image...', type=['tif'])
# Show the uploaded image to the user
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Display the uploaded image
    st.image(image)
    
    # Turn to cv2, turn from RGB to BGR and resize
    cv2_image = np.array(image)[:, :, ::-1]
    new_width = 512
    new_height = 512
    r_image = cv2.resize(cv2_image, dsize = (new_height, new_width), interpolation = cv2.INTER_LANCZOS4)
    X = np.zeros((1, new_height, new_width, 3), dtype=np.uint8)
    X[0] = r_image

# Detect mask

if st.button('Detect'):
    # Check if an image has been uploaded
    if uploaded_file is not None:
        # Check if the uploaded file is of type 'tif'
        if uploaded_file.type == 'image/tiff':
            # Predict
            # st.write('valid image')
            predicted_crack = np.where(new_model.predict(X)>0.5, 255, 0)
            st.write('Predicted crack below:')
            st.image(predicted_crack, caption='Predicted Crack')
        else:
            st.write('Please upload a TIF file')
    else:
        st.write('Please upload an image')

