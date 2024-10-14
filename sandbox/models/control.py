#!/usr/bin/env python

import datetime
import tensorflow as tf

def get_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(64,)),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(5, activation="softmax")
    ])

def load_model(model_fn: str):
    return tf.keras.models.load_model(model_fn)

def train_model(dataset_fn: str, model_fn: str, epochs=10, batch_size: int = 4):
    ds = tf.data.Dataset.load(dataset_fn)

    val_size = int(len(ds) * 0.2)
    ds_training = ds.skip(val_size)
    ds_validate = ds.take(val_size)

    ds_training = ds_training.cache().shuffle(buffer_size=100).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    ds_validate = ds_validate.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    model = get_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        ds_training,
        validation_data=ds_validate,
        epochs=epochs,
        callbacks=[tensorboard_callback]
    )

    model.save(model_fn)
