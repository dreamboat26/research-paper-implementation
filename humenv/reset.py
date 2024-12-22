# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import mujoco
from typing import Tuple

# Number of elements in qpos and qvel for the SMPL skeleton
QPOS_LEN_FOR_SMPL = 76
QVEL_LEN_FOR_SMPL = 75


def tpose(model: mujoco.MjModel, data: mujoco.MjData, random: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    qpos = np.zeros_like(data.qpos)
    qvel = np.zeros_like(data.qvel)
    qpos[2] = 0.94

    # Only reset human model position. Keep other objects as defined in the XML
    qpos[QPOS_LEN_FOR_SMPL:] = data.qpos[QPOS_LEN_FOR_SMPL:]
    qvel[QVEL_LEN_FOR_SMPL:] = data.qvel[QVEL_LEN_FOR_SMPL:]

    z_rot = 0
    euler = np.array([90, 0, z_rot])
    rad = euler * np.pi / 180
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, rad, "XYZ")
    qpos[3] = quat[0]
    qpos[4:7] = quat[1:]
    return qpos, qvel


def fall(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    random: np.random.RandomState,
    action_size: int,
    integration_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Only reset human model position. Keep other objects as defined in the XML
    data.qpos[:QPOS_LEN_FOR_SMPL] = 0
    data.qvel[:QVEL_LEN_FOR_SMPL] = 0
    z = 1
    orientation = random.random(4)
    data.qpos[2] = z
    data.qpos[3:7] = np.array(orientation)
    mujoco.mj_forward(model, data)
    n_steps = random.integers(0, 5, 1).item()
    for _ in range(n_steps):
        action = (random.random(action_size) - 0.5) * 1
        data.ctrl[:action_size] = action
        mujoco.mj_step(model, data, integration_steps)
    return data.qpos, data.qvel


def default_and_fall(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    random: np.random.RandomState,
    fall_prob: float,
    action_size: int,
    integration_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < fall_prob:
        return fall(model, data, random, action_size, integration_steps)
    else:
        return tpose(model, data, random)
