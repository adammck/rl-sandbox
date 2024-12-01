#!/usr/bin/env python

import json
import os
import time
from typing import List
import numpy as np
import grpc
from proto.gen import collector_pb2
from proto.gen import collector_pb2_grpc
from sandbox import utils

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
        self._last_action = (None, None) # d, action
        self._output = "data/capture/examples.jsonl" # todo: use param
        self._chan = self.channel = grpc.insecure_channel("localhost:50051") # todo: use param
        self._stub = collector_pb2_grpc.CollectorStub(self._chan)

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

        # if the input to the collector is the same as the previous action, then
        # don't bother asking -- we assume that the operator will provide the
        # same answer.
        if self._last_action[0] is not None:
            if self._last_action[0] == d:
                action = self._last_action[1]
                print(f"repeating last action:{action}")
                return action


        print("collecting...")

        # send the vision data to collector, and wait for a human to decide
        # which action to take.
        res = self._stub.Collect(collector_pb2.Request(
            inputs=[
                collector_pb2.Input(
                    grid=collector_pb2.Grid(rows=8, cols=8),
                    data=collector_pb2.Data(ints=collector_pb2.Ints(values=d))
                )
            ],
            # todo: derive from self.ACTIONS
            output=collector_pb2.OutputSchema(option_list=collector_pb2.OptionListSchema(options=[
                collector_pb2.Option(label="stop", hotkey=" "),              # stop
                collector_pb2.Option(label="forwards", hotkey="ArrowUp"),    # forwards
                collector_pb2.Option(label="backwards", hotkey="ArrowDown"), # backwards
                collector_pb2.Option(label="left", hotkey="ArrowLeft"),      # left
                collector_pb2.Option(label="right", hotkey="ArrowRight"),    # right
            ]))
        ))

        if not res.output.HasField('option_list'):
            raise RuntimeError("response had no option_list!")

        action = res.output.option_list.index
        print(f"action:{action}")

        # append the input and action to the training data. we convert it to a
        # tensorflow dataset later.
        with open(self._output, "a") as f:
            line = {"input": d, "output": utils.index_to_onehot(len(self.ACTIONS), action)}
            json.dump(line, fp=f)
            f.write("\n")

        # cache the result for next time
        self._last_action = (d, action)

        return action

    def _wait_for_line():
        pass
   
    def apply(self, d, action) -> None:
        act = self.ACTIONS[action]
        for i, c in enumerate(act):
            d.ctrl[i] = c
