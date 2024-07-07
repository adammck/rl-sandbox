#!/usr/bin/env python

# mostly from: https://colab.research.google.com/drive/1Hr7RQ8UatKKF7XXlHd6-7cSW0AYvN7t_

import os
import mujoco
import tensorflow as tf
import numpy as np
from PIL import Image
from sandbox import arena, utils

class TheDataset:
    def __init__(self, n):
        self._n = n

    def _generate(self):
        skipped = 0

        # keep looping until we have generated a usable example. this is a hack,
        # but is easier than fixing the generator.
        while True:

            # but don't loop forever, in case i broke the generator.
            if skipped >= 1_000:
                raise RuntimeError("skipped too many times. is the randomizer broken?")

            aren = arena.Arena()

            # step once, to initialize everything.
            mujoco.mj_forward(aren.model, aren.data)

            try:
                x, y = utils.pixel_position(aren.model, aren.data, aren.camera().id, aren.target().id, 512, 512)
            except ValueError:
                skipped += 1
                continue

            # if the target is out of view, skip this iteration.
            # TODO: move this check into pixel_position.
            if x < 0 or x >= 512 or y < 0 or y >= 512:
                skipped += 1
                continue

            pixels = aren.render()
            tfpixels = tf.expand_dims(pixels, axis=0)

            output = utils.position_to_onehot(x, y)
            tfoutput = tf.expand_dims(output, axis=0)

            return (tfpixels, tfoutput)

    def __call__(self):
        for _ in range(self._n):
            yield self._generate()

    def shape(self):
        return (
            tf.TensorSpec(shape=(1, 512, 512, 3), dtype=tf.uint8), # frame
            #tf.TensorSpec(shape=(2), dtype=tf.uint16) # (x,y) labels
            tf.TensorSpec(shape=(1, 64,), dtype=tf.uint8) # cell label
        )

gen = TheDataset(5)
os.makedirs('data/img', exist_ok=True)
for eg in gen():

    # unpack the tensor to make below readable. the batch_size dimension makes
    # this pretty awkward.
    pixels = tf.reshape(eg[0], (512, 512, 3)).numpy()
    cell = utils.onehot_to_int(tf.reshape(eg[1], (64,)).numpy())
    #x = eg[1][0]
    #y = eg[1][1]

    # don't train with the crosshair! it's just to check the x,y labels.
    #pixels = utils.add_crosshair(pixels, x, y)
    #pixels = utils.add_cell_box(pixels, cell)
    img = Image.fromarray(pixels)

    # write to disk.
    #img.save(f"data/img/x{x}_y{y}.png")
    img.save(f"data/img/cell_{cell}.png")


gen = TheDataset(1000)
ds = tf.data.Dataset.from_generator(gen, output_signature=gen.shape())
tf.data.Dataset.save(ds, "data/tfds")
