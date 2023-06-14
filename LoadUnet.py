# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:18:40 2023

@author: barbosl2, osjostedt
"""
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt

import numpy as np
# import cv2
import os
import random

import tensorflow as tf
from sklearn.model_selection import train_test_split

import tensorflow.keras.backend as K

models_path = 'Finished Models'
model_filename = 'Unet_1000epochs.h5'
histogram_equalization_key = False
continue_learning = False
generate_images_key = True

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

new_model = load_model(f'{models_path}/{model_filename}', custom_objects={'f1_score': f1_score})

## Plot model
# tf.keras.utils.plot_model(
#     new_model,
#     to_file="model.png",
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=True,
#     # show_trainable=False,
# )


# Read clean DF to output predicted images
inputs_dir = 'Inputs/'
outputs_dir = 'Outputs/'
savefigs_path = f'{outputs_dir}/Unet/{model_filename.split(".")[0]}'
seed = random.seed(1)

IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3

# Use train_test_split on all masked data
TRAIN_PATH = f'{inputs_dir}/images/'
MASKS_PATH = f'{inputs_dir}/masks/'
image_ids = os.listdir(TRAIN_PATH)

image_ids = [x for x in os.listdir(TRAIN_PATH) if x.rsplit('.')[-1] == 'tif']

# Create placeholder for train data. Tensor with length of the number of training images
X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH,
             IMG_CHANNELS), dtype=np.uint8)
Y = np.zeros((len(X), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)

# Preparing train dataset

# Read inputs: Images and masks
for image_id in range(len(image_ids)):
    X[image_id] = cv2.imread(f'{TRAIN_PATH}/{image_ids[image_id]}')
    if histogram_equalization_key:# Convert to YCrCb, split channels, apply histogram equaliation, merge back channels
        X[image_id] = cv2.cvtColor(X[image_id], cv2.COLOR_BGR2YCrCb)
        Y_channel, Cr, Cb = cv2.split(X[image_id])
        Y_channel = cv2.equalizeHist(Y_channel)
        X[image_id] = cv2.merge((Y_channel, Cr, Cb))
        X[image_id] = cv2.cvtColor(X[image_id], cv2.COLOR_YCrCb2BGR)
    
    Y_ = cv2.imread(f'{MASKS_PATH}/{image_ids[image_id]}')
    Y_ = cv2.cvtColor(Y_, cv2.COLOR_BGR2GRAY)
    Y_ = np.expand_dims(Y_, axis=-1)
    Y[image_id] = Y_

# Train, test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.05, random_state = 1)

X_test_pred = new_model.predict(X_test, verbose=1)
X_test_pred = np.where(X_test_pred>0.5, 255, 0)

X_train_pred = new_model.predict(X_train, verbose=1)
X_train_pred = np.where(X_train_pred>0.5, 255, 0)

def display(display_list, save=False, path=''):
    plt.figure(figsize=(15, 15))
    titles = ["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]), cmap='gray')
        plt.axis("off")

    if save:
        plt.savefig(f'{path}')
    plt.show()

# img_num = 58
# xtrain1 = X_train[img_num]
# ytrain1 = Y_train[img_num]
# xtrainpred = X_train_pred[img_num]

# cv2.imshow('X', xtrain1)
# cv2.imshow('Y', ytrain1)
# cv2.imshow('Pred', xtrainpred)
# cv2.waitKey()
# cv2.destroyAllWindows()

if generate_images_key:
    # Display and save figures of training data
    for img_num in range(len(X_train)):
        display([X_train[img_num], Y_train[img_num], X_train_pred[img_num]],
                save=True, path=f'{savefigs_path}/train/{img_num}.png')
    # Display and save figures of test data
    for img_num in range(len(X_test_pred)):
        display([X_test[img_num], Y_test[img_num], X_test_pred[img_num]],
                save=True, path=f'{savefigs_path}/test/{img_num}.png')

# for i in range(len(X_test_pred)):
#     cv2.imshow('test image, not used for training', np.where(X_test_pred[i]>0.5, 255, 0))
#     cv2.waitKey()
test = X_train_pred[188]
