#!/usr/bin/env python

import os
import mujoco
from sandbox import arena, utils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from PIL import Image

def generate_dataset(n: int, path:str=""):
    os.makedirs(path, exist_ok=True)

    gen = Generator(n)
    ds = tf.data.Dataset.from_generator(gen, output_signature=gen.shape())
    tf.data.Dataset.save(ds, path)

def generate_files(n: int, path="data/img"):
    os.makedirs(path, exist_ok=True)

    gen = Generator(n)
    for eg in gen():

        # unpack the tensor to make below readable.
        pixels = eg[0]
        x, y = utils.onehot_to_pos(eg[1])

        # don't train with the crosshair! it's just to check the x,y labels.
        #pixels = utils.add_crosshair(pixels, x, y)
        #pixels = utils.add_cell_box(pixels, cell)
        img = Image.fromarray(pixels)

        # write to disk.
        img.save(f"{path}/x{x}y{y}.png")

class Generator:
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
            output = utils.pos_to_onehot(x, y)
            return (pixels, output)

    def __call__(self):
        for _ in range(self._n):
            yield self._generate()

    def shape(self):
        return (
            tf.TensorSpec(shape=(512, 512, 3), dtype=tf.uint8), # frame
            tf.TensorSpec(shape=(64,), dtype=tf.uint8) # 8x8 onehot
        )