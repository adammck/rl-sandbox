#!/usr/bin/env python

import math
import random
import tempfile
from typing import Tuple

import mujoco
import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape

class Arena:
    def __init__(self, num_obstacles=5):
        env = Environment(
            loader=PackageLoader("arena", "templates"),
            autoescape=select_autoescape())
        self._tmpl = env.get_template("one.xml")
        self._num_obstacles = num_obstacles

    def xsize(self):
        return 18

    def ysize(self):
        return 18
    
    def robot(self):
       return self._model.body("robot")
    
    def camera(self):
       return self._model.camera("pov")

    def target(self):
       return self._model.body("target_red")

    def vars(self):
        return {
            "robot": Robot(self),
            "target": Target(self),
            "obstacles": [
                Obstacle(self, str(n))
                for n in range(self._num_obstacles)
            ]
        }

    def get(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        xml = self._tmpl.render(**self.vars())
        print(xml)
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(xml.encode())
            self._model = mujoco.MjModel.from_xml_path(tmp.name)
            self._data = mujoco.MjData(self._model)

        return self._model, self._data

# TODO: these are all rather duplicative. use a superclass or something.

class Robot():
    def __init__(self, arena):
        x = arena.xsize()/2
        y = arena.ysize()/2
        self._x = random.uniform(-x, x)
        self._y = random.uniform(-y, y)

    @property
    def pos(self):
        return f"{self._x} {self._y} 0"

class Target():
    def __init__(self, arena):
        x = arena.xsize()/2
        y = arena.ysize()/2
        self._x = random.uniform(-x, x)
        self._y = random.uniform(-y, y)

    @property
    def pos(self):
        return f"{self._x} {self._y} 0"

class Obstacle():
    def __init__(self, arena, name):
        ax = arena.xsize()/2
        ay = arena.ysize()/2

        # name doesn't really matter
        self._name = name

        # position
        self._x = random.uniform(-ax, ax)
        self._y = random.uniform(-ay, ay)
        self._z = 0

        # orientation (only z axis)
        self._yaw = np.radians(random.randint(0, 360))

        # size
        self._xysize = random.randrange(10, 50) / 100
        self._zsize = random.randrange(10, 20)/ 100

        # color
        self._rgbarand = random.random()
        self._luminosity = random.randrange(40, 100) / 100

    @property
    def name(self):
        return self._name

    @property
    def pos(self):
        return f"{self._x:0.2f} {self._y:0.2f} {self._z:0.2f}"

    @property
    def quat(self):
        return f"{np.cos(self._yaw/2):0.2f} 0 0 {np.sin(self._yaw/2):0.2f}"
    
    @property
    def size(self):
        return f"{self._xysize:0.2f} {self._xysize:0.2f} {self._zsize:0.2f}"

    def _round_up(self, n, decimals=0):
        multiplier = 10 ** decimals
        return math.ceil(n * multiplier) / multiplier

    @property
    def zsize2(self):
        z = self._round_up(self._zsize/2, 2)
        return str(z)

    @property
    def rgba(self):
        if self._rgbarand < 0.25:
            return f"0.2 {self._luminosity:0.2f} 0.2 1" # green
        elif self._rgbarand < 0.5:
            return f"0.2 0.2 {self._luminosity:0.2f} 1" # blue
        elif self._rgbarand < 0.75:
            return f"{self._luminosity:0.2f} {self._luminosity:0.2f} 0.2 1" # yellow
        else:
            return f"{self._luminosity:0.2f} 0.2 {self._luminosity:0.2f} 1" # purple
