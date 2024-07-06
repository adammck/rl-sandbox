#!/usr/bin/env python

import time
import mujoco
import mujoco.viewer
import arena
import controllers

a = arena.Arena()
m, d = a.get()

ctrl = controllers.Cheater(a.camera().id, a.target().id)

with mujoco.viewer.launch_passive(m, d) as viewer:
  while viewer.is_running():
    start = time.time()
    mujoco.mj_step(m, d)

    # apply changes to the viewer.
    viewer.sync()

    # if done, then stop.
    if ctrl.done(d):
       print("Done!")
       break

    # otherwise, apply the next action.
    # TODO: for the actual model, this should take pixels, not the actual data!
    ctrl.act(d)

    # wait until its time to advance.
    # this is pasted from the examples and is NOT accurate.
    time_until_next_step = m.opt.timestep - (time.time() - start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
