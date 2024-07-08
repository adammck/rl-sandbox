#!/usr/bin/env python

import os
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
from sandbox import utils

# shut up about my cpu, tensorflow
# https://github.com/tensorflow/tensorflow/issues/61688#issuecomment-1700451209
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="path to model")
    parser.add_argument("images", type=str, nargs="+", help="path to images")
    args = parser.parse_args()

    for image_fn in args.images:
        image = Image.open(image_fn)
        pixels = np.asarray(image)

        if pixels.shape != (512, 512, 3):
            parser.error("only 512x512x3 images are supported.")

        model = tf.keras.models.load_model(args.model)

        # output of the model the probabilities of each of the 64 cells. convert
        # it back into a cell index, and then the x, y pos (in cells, not px).
        probs = model.predict(np.expand_dims(pixels, axis=0))
        x, y = utils.index_to_pos(np.argmax(probs))

        print(f"{image_fn}: x={x}, y={y}")

        utils.add_box_at_pos(image, x, y)
        image.show()

if __name__ == "__main__":
    main()
