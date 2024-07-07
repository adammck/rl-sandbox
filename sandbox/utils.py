#!/usr/bin/env python

import numpy as np
from typing import List, Tuple

def onehot_to_pos(arr: List[int]) -> Tuple[int, int]:
    index = np.argmax(arr)
    y = (index // 8)
    x = (index % 8)
    return x, y

def pos_to_onehot(x: int, y: int):
    arr = np.zeros(64)
    arr[(int(y/64)*8) + int(x/64)] = 1
    return arr

def pixel_position(model, data, cam_id, target_id, width, height):

    # get the position and orientation of the camera and the robot
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3,3)
    target_pos = data.xpos[target_id]

    target_in_cam = np.dot(cam_mat.T, target_pos - cam_pos)

    if target_in_cam[2] > 0:
        raise ValueError("target is behind the camera")

    fov = model.cam_fovy[cam_id] # e.g. 45
    near = model.vis.map.znear # near clipping plane
    far = model.vis.map.zfar # far clipping plane
    aspect_ratio = width / height

    # focal scaling
    # TODO: why does this only mention height? no width?
    focal_scale = (1.0 / np.tan(np.radians(fov)/2)) * height/2.0

    # principal point (center of image)
    ppx = width / 2
    ppy = height / 2

    # matrix to project a point in cam space into 2d
    projection = np.array([
        [focal_scale / aspect_ratio, 0, 0, 0],
        [0, focal_scale, 0, 0],
        [0, 0, (far + near) / (near - far), 2.0 * (far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

    target_in_image = np.dot(projection, np.append(target_in_cam, 1.0))
    px = (target_in_image[0] / target_in_image[3])
    py = (target_in_image[1] / target_in_image[3])

    #print(f"px: {px}, py: {py}")
    return int(ppx+px), int(ppy-py)

def add_crosshair(img, x, y, s=2, color=(255, 255, 255)):

    # clamp to boundaries
    x1 = max(0, x - s)
    x2 = min(img.shape[1], x + s + 1)
    y1 = max(0, y - s)
    y2 = min(img.shape[0], y + s + 1)

    # draw two lines. seems a bit easier than diagonals.
    img[y, x1:x2] = color  # horizontal line
    img[y1:y2, x] = color  # vertical line

    return img
