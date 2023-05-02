# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:48:04 2023

@author: barbosl2
"""
# Various image resizings and plots

# Resize images to 512 x 512
# Resize masks to 512 x 512


import numpy as np
import cv2
import pandas as pd
import os
from skimage import img_as_ubyte


# Read clean DF
inputs_dir = 'Inputs/RandomForest/masks'
outputs_dir = 'Outputs/treated_masks'

images = os.listdir(inputs_dir)
images2 = []
for i in range(len(images)):
    if images[i].split('.')[-1] in ['tif']:
        images2.append(images[i])
        
images = images2

# See if the mask is actually binary 
pixel_values = []
for image in images:
    mask = cv2.imread(f'{inputs_dir}/{image}')
    flat_mask = mask.reshape(-1)
    pixel_values.append(np.unique(flat_mask).tolist())
pixels_dict = dict(zip(images, pixel_values))

# We see that many masks are not actually binary. Must convert pixel values in order to binarize
# All less than 128 turn to zero, >= 128 turn to 255

for image_name in pixels_dict.keys():
    mask = cv2.imread(f'{inputs_dir}/{image_name}')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Resize masks to 
    r_mask = cv2.resize(mask, dsize = (512, 512), interpolation = cv2.INTER_NEAREST)
    np.unique(r_mask)
    bin_mask = img_as_ubyte(np.where(r_mask>128, 255, 0))
    cv2.imwrite(f'{outputs_dir}/{image_name}', r_mask)

for image_name in pixels_dict.keys():
    img = cv2.imread(f'{inputs_dir}/{image_name}')
    r_image = cv2.resize(img, dsize = (512, 512), interpolation = cv2.INTER_LANCZOS4)
    cv2.imwrite(f'Outputs/resized_images/{image_name}', r_image)
    
# flat_mask_copy.reshape(mask.shape)
            

##### Commented out code to convert masks from jpeg to .tif
# print("importing .jpg masks")
# for i in range(51, 101):
#     print(f'{i}')
#     img = cv2.imread(f'{inputs_dir}/masks/{i}.jpg') # Original image
#     cv2.imwrite(f'{outputs_dir}/masks/{str(i)}.tif', img)


# # Check if the converted images are 0 and 255 values only:
# for i in os.listdir(f'{outputs_dir}/masks/'):
#     img = cv2.imread(f'{outputs_dir}/masks/{i}')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Read all channels or turn to gray
#     img = img.reshape(-1)
#     print(f'{i} - {np.unique(img)}')
    