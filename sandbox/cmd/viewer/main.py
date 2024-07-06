#!/usr/bin/env python

import time
import mujoco
import mujoco.viewer
from sandbox import arena, controllers

# signals
reset = False
exit = False

def keypress(code: int):
    #print(f"keypress: code={code}")
    global reset, exit

    if code == 69: # e: exit
        exit = True
    elif code == 82: # r: reset
        reset = True

while True:
   
    aren = arena.Arena()
    m = aren.model
    d = aren.data
    ctrl = controllers.Cheater(aren.camera().id, aren.target().id)

    with mujoco.viewer.launch_passive(m, d, key_callback=keypress) as viewer:

        # set the view to the default camera for this arena, so we see what the
        # robot sees.
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = aren.camera().id

        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(m, d)

            # if done, then stop.
            if reset or exit or ctrl.done(d):
                reset = False
                break

            # apply changes to the viewer.
            viewer.sync()

            # otherwise, apply the next action.
            # TODO: for the actual model, this should take pixels, not the actual data!
            with viewer.lock():
                ctrl.act(d)

            # wait until its time to advance.
            # this is pasted from the examples and is NOT accurate.
            time_until_next_step = m.opt.timestep - (time.time() - start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if exit:
        break
