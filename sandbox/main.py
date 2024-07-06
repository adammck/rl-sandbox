#!/usr/bin/env python

import time
import mujoco
import mujoco.viewer
import arena
import controllers

# signals
reset = False
exit = False

def keypress(code: int):
    global reset, exit
    print(f"keypress: code={code}")

    if code == 69: # e: exit
        exit = True
    elif code == 82: # r: reset
        reset = True

while True:
   
    aren = arena.Arena()
    m, d = aren.get()
    ctrl = controllers.Cheater(aren.camera().id, aren.target().id)

    with mujoco.viewer.launch_passive(m, d, key_callback=keypress) as viewer:
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
            ctrl.act(d)

            # wait until its time to advance.
            # this is pasted from the examples and is NOT accurate.
            time_until_next_step = m.opt.timestep - (time.time() - start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if exit:
        break
