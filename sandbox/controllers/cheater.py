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

    GOAL_DISTANCE = 1
    GOAL_ANGLE = 0.5 # deg

    ACTIONS = [
        (0, 0), # stop
        (1, 0), # forwards
        (-1, 0), # backwards
        (0, -0.5), # left
        (0, 0.5), # right
    ]

    def __init__(self, cam_id: int, target_id: int):
       self._cam_id = cam_id
       self._target_id = target_id

    def done(self, d: mujoco.MjData) -> bool:
        """Returns true if the goal has been reached."""

        dist, deg = self._get_target_dist_deg(d)
        return dist < self.GOAL_DISTANCE and (-self.GOAL_ANGLE < deg < self.GOAL_ANGLE)

    def act(self, d) -> None:
        """Applies the next action based on the given state."""

        act_id = self._action_id(d)
        self._apply_action(d, act_id)

    def _action_id(self, d: mujoco.MjData) -> int:
        dist, deg = self._get_target_dist_deg(d)
        
        # this has to be tuned so that the controller doesn't move towards a target it
        # can't see. more distant targets can be seen at a greater angle.
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

    def _get_target_dist_deg(self, d: mujoco.MjData) -> Tuple[float, float]:

        # calculate distance between robot and target.
        pr = d.xpos[self._cam_id]
        pt = d.xpos[self._target_id]
        dist = mujoco.mju_dist3(pr, pt)

        # get the direction vector in the global frame
        dir_vec = pt - pr
        dir_vec_nom = dir_vec / np.linalg.norm(dir_vec)

        # get the rotation matrix of the camera
        rot = d.xmat[self._cam_id].reshape((3, 3))

        # transform the direction vector to the camera's local frame
        dir_vec_local = np.dot(rot.T, dir_vec_nom)

        # calculate the angle in the camera's local frame
        angle = np.arctan2(dir_vec_local[1], dir_vec_local[0]) # angle relative to the camera's forward direction
        deg = np.degrees(angle)

        # HACK: why is the output wrong by 90 deg :|
        deg = 90-deg

        return dist, deg
