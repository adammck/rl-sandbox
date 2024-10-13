#!/usr/bin/env python

import time
import mujoco
import mujoco.viewer
from PIL import Image
import numpy as np
from sandbox import arena, controllers
from sandbox import utils

# signals

class Capturer:
    SECONDS_PER_ACTION = 0.5

    def __init__(self, vision_model):
        self.vision_model = vision_model
        self.arena = arena.Arena(num_targets=1, num_obstacles=0)
        self.renderer = mujoco.Renderer(self.arena.model, 512, 512)
        self.ctrl = controllers.Collector()
        #self.ctrl = controllers.Keyboard()
        #self.ctrl = controllers.Cheater(self.arena)
        self.want_exit = None
        self._next_action = None
    
    def key_callback(self, code: int):
        #print(f"key_press: {code}")

        if code == 69: # e: exit
            self.want_exit = True
            return

        return self.ctrl.key_callback(code)

    def sim_step(self, viewer) -> bool:
        """
        Returns true if the viewer should exit.
        """

        mujoco.mj_step(self.arena.model, self.arena.data)

        # apply changes to the viewer.
        viewer.sync()

        # if done (target reached), then stop.
        if self.arena.is_done():
            return True
        
        # if exit requested via keypress, also stop.
        if self.want_exit:
            return True

        # if enough time has passed, compute and apply the next action.
        if self._next_action is None or self._next_action <= time.time():
            self.action_step(viewer)
            self._next_action = time.time() + self.SECONDS_PER_ACTION

        # wait until its time to advance to the next sim step.
        # this is pasted from the examples and is NOT accurate.
        time_until_next_step = self.arena.model.opt.timestep
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
        return False

    def action_step(self, viewer):
        
        # fetch the current view as pixels.
        self.renderer.update_scene(self.arena.data, self.arena.camera().id)
        pixels = self.renderer.render()
        #Image.fromarray(pixels, 'RGB').show()

        # run the vision model over the pixels, to find the features.
        probs = self.vision_model.predict(np.expand_dims(pixels, axis=0), verbose=0)
        x, y = utils.index_to_pos(np.argmax(probs))

        # compute the next action.
        # (this may block indefinitely!)
        action = self.ctrl.next_action(probs)

        # log. this is the training data for the control model.
        print(f"x={x}, y={y} -> act={action}")

        # apply the action. this has to be a separate step, since we need to
        # acquire the lock before touching the simulator, which we don't want
        # to hold while computing the action.
        with viewer.lock():
            self.ctrl.apply(self.arena.data, action)


    def launch(self):
        self.want_exit = False
        with mujoco.viewer.launch_passive(self.arena.model, self.arena.data, key_callback=self.key_callback) as viewer:

            # set the view to the default camera for this arena, so we see what
            # the robot sees.
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = self.arena.camera().id

            while viewer.is_running():
                should_exit = self.sim_step(viewer)
                if should_exit:
                    break

def launch(vision_model):
    return Capturer(vision_model).launch()
