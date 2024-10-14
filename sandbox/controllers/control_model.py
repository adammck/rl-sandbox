#!/usr/bin/env python

import numpy as np

class ControlModel:

    def __init__(self, model):
        self._model = model

    def key_callback(self, code: int):
        pass

    def next_action(self, probs):

        # for now, cast the probs to a single int. this is the single grid cell
        # where we think the red box is. the accuracy is... okay.
        d = [0 for _ in probs[0]]
        m = np.max(probs)
        if m > (1/64)*4:
            i = np.argmax(probs)
            d[i] = 1
            print(f"control: d:{d}")
        else:
            print(f"control: no input")

        # ask the model what action we should take.
        probs = self._model.predict(np.expand_dims(d, axis=0), verbose=0)
        print(f"control: probs:{d}")
        return np.argmax(probs)

    # TODO: Move this up to the simulator; it doesn't belong here!
    ACTIONS = [
        (0, 0), # stop
        (1, 0), # forwards
        (-1, 0), # backwards
        (0, -0.5), # left
        (0, 0.5), # right
    ]

    def apply(self, d, action) -> None:
        act = self.ACTIONS[action]
        for i, c in enumerate(act):
            d.ctrl[i] = c
