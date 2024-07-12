#!/usr/bin/env python

import math
from typing import Tuple

import mujoco
import numpy as np

class Cheater:
    """
    This controller cheats, by inspecting the actual state of the world and then
    deciding where to turn. It tries to avoid doing things which an agent could
    not, but is very incomplete.
    """

    ACTIONS = [
        (0, 0), # stop
        (1, 0), # forwards
        (-1, 0), # backwards
        (0, -0.5), # left
        (0, 0.5), # right
    ]

    def __init__(self, arena):
       self._arena = arena

    def act(self, d) -> None:
        """Applies the next action based on the given state."""

        act_id = self._action_id(d)
        self._apply_action(d, act_id)

    def _action_id(self, d: mujoco.MjData) -> int:
        dist, deg = self._arena._get_target_dist_deg()
        
        # this has to be tuned so the controller doesn't move towards a target
        # # it can't see. more distant targets can be seen at a greater angle.
        deg_limit = min(20, math.pow(dist, 2)*0.25)

        if deg < -deg_limit:
            return 3 # left

        if deg > deg_limit:
            return 4 # right
        
        if dist > 0.2:
            return 1 # forwards
        
        return 0 # stop

    def _apply_action(self, d: mujoco.MjData, action_index: int):
        act = self.ACTIONS[action_index]
        for i, c in enumerate(act):
            d.ctrl[i] = c
