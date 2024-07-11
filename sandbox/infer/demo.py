#!/usr/bin/env python

import os
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
from sandbox import utils

def infer_and_show(model_fn: str, image_fn: str):
    image = Image.open(image_fn)
    pixels = np.asarray(image)

    if pixels.shape != (512, 512, 3):
        raise ValueError("only 512x512x3 images are supported.")

    model = tf.keras.models.load_model(model_fn)

    # output of the model the probabilities of each of the 64 cells. convert
    # it back into a cell index, and then the x, y pos (in cells, not px).
    probs = model.predict(np.expand_dims(pixels, axis=0))
    x, y = utils.index_to_pos(np.argmax(probs))

    print(f"{image_fn}: x={x}, y={y}")

    utils.add_box_at_pos(image, x, y)
    image.show()
