#!/usr/bin/env python

import datetime
import tensorflow as tf
from sandbox import models

ds = tf.data.Dataset.load("data/tfds")

# partition dataset
val_size = int(len(ds) * 0.2)
ds_training = ds.skip(val_size)
ds_validate = ds.take(val_size)


batch_size = 4
ds_training = ds_training.cache().shuffle(buffer_size=100).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
ds_validate = ds_validate.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

model = models.vision()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"])

run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = "logs/" + run_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
print(f"logging to {log_dir}...")

model.fit(
    ds_training,
    validation_data=ds_validate,
    epochs=10, 
    callbacks=[tensorboard_callback])

fn = f"{run_name}.keras"
model.save(fn)
print(f"saved to {fn}")
