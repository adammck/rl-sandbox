#!/usr/bin/env python

import os
import random
import mujoco
from sandbox import arena, utils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from PIL import Image

def generate_dataset(n: int, path:str="", ratio=0.2):
    os.makedirs(path, exist_ok=True)
    gen = Generator(n, ratio)
    ds = tf.data.Dataset.from_generator(gen, output_signature=gen.shape())
    tf.data.Dataset.save(ds, path)

def generate_files(n: int, path="data/img", ratio=0.2):
    os.makedirs(path, exist_ok=True)
    gen = Generator(n, ratio)
    for eg in gen():

        # unpack the tensor to make below readable.
        pixels = eg[0]

        # don't train with the crosshair! it's just to check the x,y labels.
        #pixels = utils.add_crosshair(pixels, x, y)
        #pixels = utils.add_cell_box(pixels, cell)
        img = Image.fromarray(pixels)

        if utils.onehot_is_zero(eg[1]):
            fn = "none"
        else:
            x, y = utils.onehot_to_pos(eg[1])
            fn = f"x{x}y{y}"

        # write to disk.
        rnd = random.randint(111111, 999999)
        img.save(f"{path}/{fn}-{rnd}.png")

class Generator:
    def __init__(self, n, ratio):
        self._n = int(n*0.9)
        self._m = n - self._n

    def _generate(self, num_targets=1):

        # this one can't fail.
        if num_targets == 0:
            return self._gen_without_target()
    
        # keep looping until we have generated a usable example. this is a hack,
        # but is easier than fixing the generator.
        skipped = 0
        while True:

            # but don't loop forever, in case i broke the generator.
            if skipped >= 1_000:
                raise RuntimeError("skipped too many times. is the randomizer broken?")

            try:
                return self._gen_with_target(num_targets)
            except RuntimeError:
                skipped += 1
                continue

    def _gen_with_target(self, num_targets):
        aren = arena.Arena(num_targets=num_targets)

        # step once, to initialize everything.
        mujoco.mj_forward(aren.model, aren.data)

        try:
            x, y = utils.pixel_position(aren.model, aren.data, aren.camera().id, aren.target().id, 512, 512)
        except ValueError as ex:
            raise RuntimeError(str(ex))

        # if the target is out of view, skip this iteration.
        # TODO: move this check into pixel_position.
        if x < 0 or x >= 512 or y < 0 or y >= 512:
            raise RuntimeError("target not visible")

        pixels = aren.render()
        output = utils.pos_to_onehot(x, y)
        return (pixels, output)

    def _gen_without_target(self):
        aren = arena.Arena(num_targets=0)
        mujoco.mj_forward(aren.model, aren.data)
        return (aren.render(), utils.zero_onehot())
        
    def __call__(self):
        for _ in range(self._n):
            yield self._generate(1)

        for _ in range(self._m):
            yield self._generate(0)

    def shape(self):
        return (
            tf.TensorSpec(shape=(512, 512, 3), dtype=tf.uint8), # frame
            tf.TensorSpec(shape=(64,), dtype=tf.uint8) # 8x8 onehot
        )