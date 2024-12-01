#!/usr/bin/env python

import math
import random
import tempfile
from typing import Tuple

import mujoco
import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape


class Arena:
    GOAL_DISTANCE = 1.5
    GOAL_ANGLE = 10 # deg

    def __init__(self, num_targets=1, num_obstacles=5):
        self._num_targets = num_targets
        self._num_obstacles = num_obstacles

        env = Environment(
            loader=PackageLoader("sandbox.arena", "templates"),
            autoescape=select_autoescape())
        tmpl = env.get_template("one.xml")
        xml = tmpl.render(**self._vars())

        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)

    def is_done(self) -> bool:
        """Returns true if the goal has been reached."""

        dist, deg = self._get_target_dist_deg()
        return dist < self.GOAL_DISTANCE and (-self.GOAL_ANGLE < deg < self.GOAL_ANGLE)

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    def render(self, cam_name=None) -> np.ndarray:

        if cam_name is None:
            cam_name = self.camera().name

        opt = mujoco.MjvOption()
        ren = mujoco.Renderer(self._model, width=512, height=512)
        ren.update_scene(self._data, camera=cam_name, scene_option=opt)
        pixels = ren.render()

        # render a crosshair if it's visible
        #   px, py = get_pxpy(mjmodel, mjdata, ren)
        #   if 0 <= px < ren.width and 0 <= py < ren.height:
        #     pixels = add_crosshair(pixels, px, py)

        return pixels

    def target_visible(self):
        """Returns true if the target is currently visible to the robot."""
        pass

    def xsize(self):
        return 18

    def ysize(self):
        return 18
    
    def robot(self):
       return self._model.body("robot")
    
    def camera(self, name: str="pov"):
       return self._model.camera(name)

    def target(self):
       return self._model.body("target_red")

    def _vars(self):
        if self._num_targets == 0:
            t = None

        elif self._num_targets == 1:
            t = Target(self)

        else:
            raise ValueError("only zero or one targets is currently supported")

        return {
            "robot": Robot(self),
            "target": t,
            "obstacles": [
                Obstacle(self, str(n))
                for n in range(self._num_obstacles)
            ]
        }

    def _get_target_dist_deg(self) -> Tuple[float, float]:
        cam_id = self.camera().id

        # calculate distance between robot and target.
        pr = self._data.xpos[cam_id]
        pt = self._data.xpos[self.target().id]
        dist = mujoco.mju_dist3(pr, pt)

        # get the direction vector in the global frame
        dir_vec = pt - pr
        dir_vec_nom = dir_vec / np.linalg.norm(dir_vec)

        # get the rotation matrix of the camera
        rot = self._data.xmat[cam_id].reshape((3, 3))

        # transform the direction vector to the camera's local frame
        dir_vec_local = np.dot(rot.T, dir_vec_nom)

        # calculate the angle in the camera's local frame
        angle = np.arctan2(dir_vec_local[1], dir_vec_local[0]) # angle relative to the camera's forward direction
        deg = np.degrees(angle)

        # HACK: why is the output wrong by 90 deg :|
        deg = 90-deg

        return dist, deg

# TODO: these are all rather duplicative. use a superclass or something.

class Robot():
    def __init__(self, arena):
        x = arena.xsize()/2
        y = arena.ysize()/2
        self._x = random.uniform(-x, x)
        self._y = random.uniform(-y, y)

        # orientation (only z axis)
        self._yaw = np.radians(random.randint(0, 360))

    @property
    def pos(self):
        return f"{self._x} {self._y} 0"

    @property
    def quat(self):
        return f"{np.cos(self._yaw/2):0.2f} 0 0 {np.sin(self._yaw/2):0.2f}"

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

    @property
    def zsize(self):
        return f"{self._zsize:0.2f}"

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
