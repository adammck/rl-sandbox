#!/usr/bin/env python

import tensorflow as tf

def get_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(512, 512, 3)),

        # scale 0-255 pixel values to 0-1
        tf.keras.layers.Rescaling(1./255),

        # idk lol
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        # magic goes here
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # output is an 8*8 grid. zeros except for the one which the target is in.
        tf.keras.layers.Dense(64, activation="softmax")
    ])
