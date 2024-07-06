#!/usr/bin/env python

import os
import random
from typing import Tuple

import mujoco
import numpy as np

class Arena:
    def __init__(self, fn):
       # TODO: generate the XML from the positions rather than shuffling.
       path = os.path.abspath(f"../arenas/{fn}")
       self._model = mujoco.MjModel.from_xml_path(path)
       self._data = mujoco.MjData(self._model)

    def robot(self):
       return self._model.body("robot")
    
    def camera(self):
       return self._model.camera("pov")

    def target(self):
       return self._model.body("target_red")

    def obstacles(self):
       return [
          self._model.body(name)
          for name in ["target_green", "target_blue"]
       ]

    def get(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
       self._shuffle()
       return self._model, self._data
       
    def _random_pos(self):
        yaw = np.radians(random.randint(0, 360))
        return np.array([random.uniform(-9, 9), random.uniform(-9, 9), 0, np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])

    def _shuffle(self):
        self._shuffle_one(self.robot().id)
        self._shuffle_one(self.target().id)

        for body in self.obstacles():
            self._shuffle_one(body.id)

    def _shuffle_one(self, body_id: int):
        pos = self._random_pos()
        addr = self._model.jnt_qposadr[self._model.body_jntadr[body_id]]
        self._data.qpos[addr:addr+7] = pos
       