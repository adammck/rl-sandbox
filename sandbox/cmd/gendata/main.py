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
            if x < 0 or x > 512 or y < 0 or y > 512:
                skipped += 1
                continue

            pixels = aren.render()
            return (pixels, (x, y))

    def __call__(self):
        for _ in range(self._n):
            yield self._generate()

    def shape(self):
        return (
            tf.TensorSpec(shape=(512, 512, 3), dtype=tf.uint8), # frame
            tf.TensorSpec(shape=(2), dtype=tf.uint16) # (x,y) labels
        )

gen = TheDataset(10)
os.makedirs('data/img', exist_ok=True)
for eg in gen():

    # unpack the tensor to make below readable.
    pixels = eg[0]
    x = eg[1][0]
    y = eg[1][1]

    # don't train with the crosshair! it's just to check the x,y labels.
    pixels = utils.add_crosshair(pixels, x, y)
    img = Image.fromarray(pixels)

    # write to disk.
    img.save(f"data/img/x{x}_y{y}.png")


gen = TheDataset(10)
ds = tf.data.Dataset.from_generator(gen, output_signature=gen.shape())
tf.data.Dataset.save(ds, "data/tfds")
