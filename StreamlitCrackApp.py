# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:23:03 2023

@author: osjostedt, barbosl2
"""

import numpy as np
import streamlit as st
from PIL import Image
import cv2
from keras.models import load_model
import tensorflow.keras.backend as K
import plotly.graph_objs as go
# from streamlit_plotly_events import plotly_events

# """
# Run streamlit app by running: streamlit run StreamlitCrackApp.py
# App will open in browser on a localhost
# """

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

ruler_len_mm = st.number_input('Ruler length in [mm]:', value=2.0)


if uploaded_file is not None:
    # Import image
    image = Image.open(uploaded_file)
    img_width = image.width
    img_height = image.height
    # Create the figure

    # Define the plotly chart config
    layout = go.Layout(
        images=[go.layout.Image(
            source=image,
            xref="x", yref="y",
            x=0, y=1, sizex=img_width, sizey=img_height,
            opacity=1,
            layer="below")],
        
        dragmode='drawline', # False,
        newshape=dict(line_color="red"),
        hoverdistance = 0,
        title_text='Draw the scale line to get the area of the crack',
        clickmode='event+select',
        uirevision='shapes',
        
        xaxis=dict(
            range=[0,img_width],
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            # showticklabels=False
        ),
        yaxis=dict(
            range=(img_height, 0),
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            # showticklabels=False,
            scaleanchor='x'
        )
    )
    
    config = {'displaylogo': False,
              'displayModeBar': True,
              'modeBarButtonsToRemove': ['zoom'],#, 'pan'], #, 'lasso2d'],# 'select2d'
              'modeBarButtonsToAdd':['drawline', 'eraseshape', 'lasso2d']
        }
    
    # Define the initial data for the figure
    data = [go.Scatter(x=[0], y=[0], mode='lines')]
    fig = go.Figure(data=data,layout=layout) # data=data
    
    ################### Add ruler  ###################
    
    # Add line to be used as ruler
    fig.add_shape(type = 'line', x0 = 0, x1 = 100, y0 = 0, y1 = 0,
                  xref='x', yref='y', line_color='red',
                  name='ruler', editable = True)
    fig.update_layout()
    
    st.plotly_chart(fig, config=config)
    # st.write(fig.layout.shapes)
    
    ################### Prediction ###################
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
            # st.write(fig.layout.shapes)
            with st.expander('Detected crack', expanded = True):
                # Get prediction, threshold, turn to cv2, resize back to original shape
                predicted_crack = np.where(new_model.predict(X)>0.5, 255, 0)
                # st.write(predicted_crack.shape)
                predicted_crack = cv2.resize(predicted_crack[0], dsize = (img_width, img_height), interpolation = cv2.INTER_NEAREST)
                
                st.image(predicted_crack, caption='Predicted Crack')
                st.write('Assuming 100 px on ruler, still to be implemented')
                ruler_len_px = 100 # [px]
                st.write(f'Ruler length in mm is: {ruler_len_mm}')
                # Calculate crack area by ruelr length in [px], in [mm] and crack label count
                crack_area = (ruler_len_mm/ruler_len_px)*np.unique(predicted_crack, return_counts=True)[1][0]
                st.write(f'total crack area is {crack_area:.2f} [mm^2]') # TODO limit float decimal places
        else:
            st.write('Please upload a TIF file')
    else:
        st.write('Please upload an image')
        