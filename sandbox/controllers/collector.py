#!/usr/bin/env python

import json
import os
import time
from typing import List
import numpy as np

class Collector:
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
        pass

    def next_action(self, probs: List[float]):
        print(f"probs:{probs}")

        # for now, cast the probs to a single int.
        # TODO: remove this once the collector can handle floats.
        d = [0 for _ in probs[0]]

        m = np.max(probs)
        print(f"m:{m}")
        if m > (1/64)*4:
            i = np.argmax(probs)
            d[i] = 1
        print(f"d:{d}")

        fn = "../collector/examples.jsonl"
        last_size = os.path.getsize(fn)

        # write the vision data to file so the collector can find it.
        # TODO: inject the path via command line arg
        with open("../collector/input/tmp.jsonl", "w") as f:
            json.dump(d, f)

        new_data = {}

        print("waiting...")

        # wait for a line to be added to the output file, indicating the the
        # has received the vision data, and submitted a response.
        while True:
            s = os.path.getsize(fn)
            if s == last_size:
                time.sleep(0.1)
                continue

            with open(fn, "r") as f:
                f.seek(last_size)
                line = f.readline()

            new_data = json.loads(line.strip())
            last_size = s
            break

        print(f"got data! {new_data}")

        # check that the input included in the new line matches the input that
        # we submitted. otherwise we're racing and/or confused.
        if new_data["input"] != d:
            raise RuntimeError("output line from collector didn't match input")

        output = new_data["output"]
        if len(output) != len(self.ACTIONS):
            raise RuntimeError("output length did not match actions length")

        action = np.argmax(output)
        print(f"action:{action}")
        return action

    def _wait_for_line():
        pass
   
    def apply(self, d, action) -> None:
        act = self.ACTIONS[action]
        for i, c in enumerate(act):
            d.ctrl[i] = c
