#!/usr/bin/env python

# mostly from: https://colab.research.google.com/drive/1Hr7RQ8UatKKF7XXlHd6-7cSW0AYvN7t_

import time
import mujoco
import numpy as np
from PIL import Image
from sandbox import arena, utils


examples_to_generate = 5

generated = 0
skipped = 0
while True:

    # don't loop forever.
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
    if x < 0 or x > 512 or y < 0 or y > 512:
        skipped += 1
        continue

    pixels = aren.render()
    pixels = utils.add_crosshair(pixels, x, y)
    new_image = Image.fromarray(pixels)
    new_image.show()

    # reset skip counter. it's per-attempt, not cumulative.
    skipped = 0

    # stop generating when we've got enough
    generated += 1
    if generated >= examples_to_generate:
        break
