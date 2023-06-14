# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:14:57 2023

@author: barbosl2, osjostedt
"""

# Using U-net and keras to segment the image 

# Must resize the images to keep input dimensions.
# Masks resized and treated to keep only 0 and 255 as pixel values

import numpy as np
import cv2
import os
import random 
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split

import tensorflow.keras.backend as K

# Read clean DF
inputs_dir = 'Inputs/'
outputs_dir = 'Outputs/'
train_key = True # True to train model
epochs_to_train = 10
histogram_equalization_key = False # True to do histogram equalization
seed = random.seed(1)

IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3


TRAIN_PATH = f'{inputs_dir}/images/'
MASKS_PATH = f'{inputs_dir}/masks/'

# List with all images names
image_ids = [x for x in os.listdir(TRAIN_PATH) if x.rsplit('.')[-1] == 'tif']

# Create placeholder for train data. Tensor with length of the number of training images
X =  np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y = np.zeros((len(X), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool_)

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

# Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) # Convert uint8 to float, normalizes the values

# c stands for convolution operation
# p stands for pooling layer

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)


# Define f1 score

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

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(), f1_score, tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5)])
model.summary()

########################################

# Tensorboard
train_writer = tf.summary.create_file_writer('logs/train/')
test_writer = tf.summary.create_file_writer('logs/test/')

# Model Checkpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('checkpoint.h5', verbose = 1, save_best_only = False, save_freq = 300)

callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=5, monitor = 'val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir = 'logs/', histogram_freq = 1),
    checkpointer
    ]


if train_key:    
    results = model.fit(X_train, Y_train, validation_split=0.1,
                        batch_size=16,
                        epochs=epochs_to_train, callbacks=callbacks, verbose = 1)
    model.save(f'Finished Models/Unet_{epochs_to_train}epochs.h5')
    
    #############################
    
    """
    We can use Tensorboard to keep track of the learning. 
    The callback TensorBoard will give a graphic display of the training 
    metrics at the browser. To use it, point anaconda prompt to the root 
    directory and type "tensorboard --logdir logs".
    Will host tool on localhost:6006
    """
    
    #############################
    # Postprocessing
    
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
