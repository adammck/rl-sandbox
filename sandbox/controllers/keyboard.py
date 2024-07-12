#!/usr/bin/env python

import time

class Keyboard:
    ACTIONS = [
        (0, 0), # stop
        (1, 0), # forwards
        (-1, 0), # backwards
        (0, -0.5), # left
        (0, 0.5), # right
    ]

    def __init__(self):
        self._action = 0
        self._blocking = False

    def key_callback(self, code: int):
        if code == 263: # left
            self._action = 3
        elif code == 262: # right
            self._action = 4
        elif code == 265: # up
            self._action = 1
        elif code == 264: # down
            self._action = 2
        elif code == 32: # space
            self._action = 0

    def next_action(self):
        # wait for a keypress before acting.
        # TODO: do i need a lock here? i have no idea how to python
        if self._blocking:
            self._action = 0

            while True:
                if self._action > 0: break
                time.sleep(0.01)
        
        return self._action

    def apply(self, d, action) -> None:
        act = self.ACTIONS[action]
        for i, c in enumerate(act):
            d.ctrl[i] = c
