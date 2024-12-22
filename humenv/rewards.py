# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import abc
from typing import Optional
import numpy as np
import mujoco
import re
from dm_control.utils import rewards


COORD_TO_INDEX = {"x": 0, "y": 1, "z": 2}
ALIGNMENT_BOUNDS = {"x": (-0.1, 0.1), "y": (0.9, float("inf")), "z": (-0.1, 0.1)}

# [min_bound, max_bound, margin] to be used in rewards.tolerance in arms_reward
REWARD_LIMITS = {
    "l": [0, 0.8, 0.2],
    "m": [1.4, 1.6, 0.1],
    "h": [1.8, float("inf"), 0.2],
    "x": [0, float("inf"), 1],
}


def rot2eul(R: np.ndarray):
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


def get_xpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1
    xpos = data.xpos[index].copy()
    return xpos


def get_xmat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1
    xmat = data.xmat[index].reshape((3, 3)).copy()
    return xmat


def get_chest_upright(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    chest_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Chest")
    assert chest_index > -1
    chest_upright = data.xmat[chest_index][-2]
    return chest_upright


def get_sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str):
    chest_gyro_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)  # in global coordinate
    assert chest_gyro_index > -1
    start = model.sensor_adr[chest_gyro_index]
    end = start + model.sensor_dim[chest_gyro_index]
    sensord = data.sensordata[start:end].copy()
    return sensord


def get_center_of_mass_linvel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    chest_subtree_linvel_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "Chest_subtreelinvel")  # in global coordinate
    start = model.sensor_adr[chest_subtree_linvel_index]
    end = start + model.sensor_dim[chest_subtree_linvel_index]
    center_of_mass_velocity = data.sensordata[start:end].copy()
    return center_of_mass_velocity


class RewardFunction(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float: ...

    @staticmethod
    @abc.abstractmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]: ...

    def __call__(
        self,
        model: mujoco.MjModel,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
    ):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        return self.compute(model, data)


@dataclasses.dataclass
class ZeroReward(RewardFunction):
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        return 0.0

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name.lower() in ["none", "zero", "rewardfree"]:
            return ZeroReward()
        return None


@dataclasses.dataclass
class LocomotionReward(RewardFunction):
    move_speed: float = 5
    stand_height: float = 1.4
    move_angle: float = 0
    egocentric_target: bool = True
    low_height: float = 0.6
    stay_low: bool = False

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_h = data.xpos.copy()[1:25][0, 2]
        head_height = get_xpos(model, data, name="Head")[-1]
        chest_upright = get_chest_upright(model, data)
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        if self.move_angle is not None:
            move_angle = np.deg2rad(self.move_angle - 90)  # somehow the the "front" of the humanoid is actually the left side
        if self.egocentric_target:
            chest_xmat = get_xmat(model, data, name="Chest")
            euler = rot2eul(chest_xmat)
            move_angle = move_angle + euler[-1]

        if self.stay_low:
            standing = rewards.tolerance(
                root_h,
                bounds=(self.low_height / 2, self.low_height),
                margin=self.low_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = rewards.tolerance(
                head_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        upright = rewards.tolerance(
            chest_upright,
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5
        if self.move_speed == 0:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.5).mean()
            return small_control * stand_reward * dont_move
        else:
            vel = center_of_mass_velocity[[0, 1]]
            com_velocity = np.linalg.norm(vel)
            move = rewards.tolerance(
                com_velocity,
                bounds=(
                    self.move_speed - 0.1 * self.move_speed,
                    self.move_speed + 0.1 * self.move_speed,
                ),
                margin=self.move_speed / 2,
                value_at_margin=0.5,
                sigmoid="gaussian",
            )
            move = (5 * move + 1) / 6
            # move in a specific direction
            if np.isclose(com_velocity, 0.0) or move_angle is None:
                angle_reward = 1.0
            else:
                direction = vel / (com_velocity + 1e-6)
                target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
                dot = target_direction.dot(direction)
                angle_reward = (dot + 1.0) / 2.0
            reward = small_control * stand_reward * move * angle_reward
            return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed = float(match.group(1)), float(match.group(2))
            return LocomotionReward(move_angle=move_angle, move_speed=move_speed)
        pattern = r"^move-ego-low-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed = float(match.group(1)), float(match.group(2))
            return LocomotionReward(move_angle=move_angle, move_speed=move_speed, stay_low=True)
        return None


@dataclasses.dataclass
class JumpReward(RewardFunction):

    jump_height: float = 1.6
    max_velocity: float = 5.0

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        head_height = get_xpos(model, data, name="Head")[-1]
        chest_upright = get_chest_upright(model, data)
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)

        jumping = rewards.tolerance(
            head_height,
            bounds=(self.jump_height, self.jump_height + 0.1),
            margin=self.jump_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upright = rewards.tolerance(
            chest_upright,
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        up_velocity = rewards.tolerance(
            center_of_mass_velocity[-1],
            bounds=(self.max_velocity, float("inf")),
            sigmoid="linear",
            margin=self.max_velocity,
            value_at_margin=0,
        )
        reward = jumping * upright * up_velocity
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^jump-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            jump_height = float(match.group(1))
            return JumpReward(jump_height=jump_height)
        return None


@dataclasses.dataclass
class HeadstandReward(RewardFunction):

    stand_pelvis_height: float = 0.95

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="Pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="Pelvis")
        pelvis_orientation = pelvis_xmat[2, :].ravel()
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        angular_velocity = get_sensor_data(model, data, "Pelvis_gyro")
        left_foot_h = get_xpos(model, data, name="L_Ankle")[-1]
        right_foot_h = get_xpos(model, data, name="R_Ankle")[-1]
        head_h = get_xpos(model, data, name="Head")[-1]
        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height / 2,
            value_at_margin=0.01,
            sigmoid="linear",
        )

        small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5

        headstand = rewards.tolerance(
            pelvis_orientation[COORD_TO_INDEX["y"]],
            bounds=(-1, -0.9),
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.5).mean()
        dont_rotate = rewards.tolerance(
            angular_velocity,
            bounds=(-1, 1),
            margin=4,
            value_at_margin=0.1,
            sigmoid="linear",
        ).mean()

        high_left_foot = rewards.tolerance(
            left_foot_h,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height / 2,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        high_right_foot = rewards.tolerance(
            right_foot_h,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height / 2,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        # head touches the floor at height ~0.22
        high_head = rewards.tolerance(
            head_h,
            bounds=(0.3, float("inf")),
            margin=0.1,
            value_at_margin=0.01,
            sigmoid="linear",
        )

        reward = height_reward * small_control * headstand * dont_move * dont_rotate * high_left_foot * high_right_foot * high_head
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name == "headstand":
            return HeadstandReward()
        return None


@dataclasses.dataclass
class RotationReward(RewardFunction):

    axis: str = "x"
    target_ang_velocity: float = 5.0
    stand_pelvis_height: float = 0.8

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="Pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="Pelvis")
        pelvis_orientation = pelvis_xmat[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "Pelvis_gyro")
        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)

        small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5

        targ_av = np.abs(self.target_ang_velocity)
        move = rewards.tolerance(
            direction * angular_velocity[COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )

        aligned = rewards.tolerance(
            pelvis_orientation[COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        reward = move * height_reward * small_control * aligned
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rotate-(x|y|z)-(-?\d+\.*\d*)-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            axis, target_ang_velocity, stand_pelvis_height = (
                match.group(1),
                float(match.group(2)),
                float(match.group(3)),
            )
            return RotationReward(
                axis=axis,
                target_ang_velocity=target_ang_velocity,
                stand_pelvis_height=stand_pelvis_height,
            )
        return None


@dataclasses.dataclass
class ArmsReward(RewardFunction):

    stand_height: float = 1.4
    left_pose: str = "h"
    right_pose: str = "h"

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        head_height = get_xpos(model, data, name="Head")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        left_height = get_xpos(model, data, name="L_Hand")[-1]
        right_height = get_xpos(model, data, name="R_Hand")[-1]
        chest_upright = get_chest_upright(model, data)
        standing = rewards.tolerance(
            head_height,
            bounds=(self.stand_height, float("inf")),
            margin=self.stand_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upright = rewards.tolerance(
            chest_upright,
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.5).mean()
        small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5
        return small_control * stand_reward * dont_move * left_arm * right_arm

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^raisearms-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            left_pose, right_pose = match.group(1), match.group(2)
            return ArmsReward(left_pose=left_pose, right_pose=right_pose)
        return None


@dataclasses.dataclass
class LieDownReward(RewardFunction):

    direction: str = "up"

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        chest_upright = get_chest_upright(model, data)
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        if type(self.direction) is str:  # TODO: Quick hack, this could be done once in init
            self.direction = 1 if self.direction == "up" else -1
        orientations = []
        for el in [
            "Head",
            "Torso",
            "Chest",
            "Pelvis",
            "L_Knee",
            "L_Ankle",
            "R_Knee",
            "R_Ankle",
        ]:
            _index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, el)
            x = data.xmat[_index][-3:]
            orientations.append(x)
        positions = [
            get_xpos(model, data, name="Pelvis")[-1],
            get_xpos(model, data, name="Head")[-1],
            get_xpos(model, data, name="L_Knee")[-1],
            get_xpos(model, data, name="R_Knee")[-1],
            get_xpos(model, data, name="R_Ankle")[-1],
            get_xpos(model, data, name="L_Ankle")[-1],
        ]

        rew_ground = 1
        for el in positions:
            rew_ground *= rewards.tolerance(el, bounds=(0, 0.2), sigmoid="linear", margin=0.7, value_at_margin=0)
        upright = rewards.tolerance(
            chest_upright,
            bounds=(0, 0.2),
            sigmoid="linear",
            margin=1,
            value_at_margin=0,
        )
        ground_reward = rew_ground * upright
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.5).mean()
        small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5
        orient_reward = 1
        for x in orientations:
            orient_reward *= rewards.tolerance(
                x[-1] * self.direction,
                bounds=(0.95, float("inf")),
                sigmoid="linear",
                margin=1.9,
                value_at_margin=0,
            )
        return small_control * ground_reward * dont_move * orient_reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name == "lieonground-up":
            direction = "up"
            return LieDownReward(direction=direction)
        if name == "lieonground-down":
            direction = "down"
            return LieDownReward(direction=direction)
        return None


@dataclasses.dataclass
class SplitReward(RewardFunction):

    distance: float = 1.5

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        la = get_xpos(model, data, "L_Ankle")
        ra = get_xpos(model, data, "R_Ankle")
        pelvis_z = get_xpos(model, data, "Pelvis")[-1]
        head_z = get_xpos(model, data, "Head")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        diff = np.linalg.norm(la - ra)
        split_rew = rewards.tolerance(
            diff,
            bounds=(self.distance, float("inf")),
            margin=0.5,
            value_at_margin=0,
            sigmoid="linear",
        )
        pelvis_pos = rewards.tolerance(pelvis_z, bounds=(0, 0.2), margin=0.5, value_at_margin=0, sigmoid="linear")
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.5).mean()
        small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5

        head_rew = rewards.tolerance(
            head_z,
            bounds=(0.5, float(np.inf)),
            margin=0.3,
            value_at_margin=0,
            sigmoid="linear",
        )

        return head_rew * split_rew * pelvis_pos * dont_move * small_control

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^split-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            distance = float(match.group(1))
            return SplitReward(distance=distance)
        return None


@dataclasses.dataclass
class SitOnGroundReward(RewardFunction):

    pelvis_height_th: float = 0
    constrained_knees: bool = False
    knees_not_on_ground: bool = False

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        HEAD_PELVIS_GAP = 0.58

        pelvis_height = get_xpos(model, data, name="Pelvis")[-1]
        head_height = get_xpos(model, data, "Head")[-1]
        left_knee_pos = get_xpos(model, data, name="L_Knee")[-1]
        right_knee_pos = get_xpos(model, data, name="R_Knee")[-1]
        chest_upright = get_chest_upright(model, data)
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        standing_height = self.pelvis_height_th + HEAD_PELVIS_GAP
        standing = rewards.tolerance(
            head_height,
            bounds=(standing_height, standing_height + 0.2),
            margin=0.1,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upright = rewards.tolerance(
            chest_upright,
            bounds=(0.85, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.5).mean()
        small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5
        pelvis_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.pelvis_height_th, self.pelvis_height_th + 0.15),
            sigmoid="linear",
            margin=0.7,
            value_at_margin=0,
        )
        knee_reward = 1
        if self.constrained_knees:
            knee_reward *= rewards.tolerance(
                left_knee_pos,
                bounds=(0, 0.1),
                sigmoid="linear",
                margin=0.7,
                value_at_margin=0,
            )
            knee_reward *= rewards.tolerance(
                right_knee_pos,
                bounds=(0, 0.1),
                sigmoid="linear",
                margin=0.7,
                value_at_margin=0,
            )
        if self.knees_not_on_ground:
            knee_reward *= rewards.tolerance(
                left_knee_pos,
                bounds=(0.2, 1),
                sigmoid="linear",
                margin=0.1,
                value_at_margin=0,
            )
            knee_reward *= rewards.tolerance(
                right_knee_pos,
                bounds=(0.2, 1),
                sigmoid="linear",
                margin=0.1,
                value_at_margin=0,
            )
        return small_control * stand_reward * dont_move * pelvis_reward * knee_reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name == "sitonground":
            pelvis_height_th = 0
            return SitOnGroundReward(pelvis_height_th=pelvis_height_th, constrained_knees=True)
        pattern = r"^crouch-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            pelvis_height_th = float(match.group(1))
            return SitOnGroundReward(pelvis_height_th=pelvis_height_th, knees_not_on_ground=True)
        return None


@dataclasses.dataclass
class CrawlReward(RewardFunction):

    spine_height: float = 0.3
    move_angle: float = 0
    move_speed: float = 1
    direction: float = -1

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        move_angle = np.deg2rad(self.move_angle - 90)
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)

        spine_high_reward = []
        orientation_reward = []
        for el in ["Spine", "Torso", "Chest", "Pelvis", "Head"]:
            _index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, el)
            z = data.xpos[_index][-1]
            up_bound = self.spine_height + 0.2
            down_bound = self.spine_height
            if el == "Head":
                up_bound = 1
                down_bound = 0.3
            spine_high_reward.append(
                rewards.tolerance(
                    z,
                    bounds=(down_bound, up_bound),
                    margin=0.1,
                    value_at_margin=0.01,
                    sigmoid="linear",
                )
            )
            x = data.xmat[_index][-3:]
            orientation_reward.append(
                rewards.tolerance(
                    x[-1] * self.direction,
                    bounds=(0.5, float("inf")),
                    margin=0.5,
                    value_at_margin=0,
                    sigmoid="linear",
                )
            )

        for el in ["L_Knee", "R_Knee", "L_Hip", "R_Hip", "L_Ankle", "R_Ankle"]:
            _index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, el)
            x = data.xmat[_index][-3:]
            orientation_reward.append(
                rewards.tolerance(
                    x[-1] * self.direction,
                    bounds=(0.5, float("inf")),
                    margin=0.5,
                    value_at_margin=0,
                    sigmoid="linear",
                )
            )
        pos_orient_reward = np.prod(orientation_reward) * (1.0 + np.prod(spine_high_reward)) / 2.0

        # velocity
        _index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "Chest")
        xmat = data.xmat[_index].reshape((3, 3)).copy()
        chest_euler = rot2eul(xmat)
        move_angle = move_angle + chest_euler[-1]

        angle_alignment = []
        for el in ["Spine", "Torso", "Chest", "Pelvis"]:
            _index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, el)
            xmat = data.xmat[_index].reshape((3, 3)).copy()
            euler = rot2eul(xmat)
            angle_alignment.append(
                rewards.tolerance(
                    euler[-1],
                    bounds=(chest_euler[-1] - 0.1, chest_euler[-1] + 0.1),
                    margin=0.5,
                    value_at_margin=0,
                    sigmoid="linear",
                )
            )
        angle_alignment = (1 + np.prod(angle_alignment)) / 2.0

        chest_gyro_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "Pelvis_gyro")  # in global coordinate
        start = model.sensor_adr[chest_gyro_index]
        end = start + model.sensor_dim[chest_gyro_index]
        angular_velocity = data.sensordata[start:end].copy()
        dont_rotate = rewards.tolerance(
            np.abs(angular_velocity),
            bounds=(0, 2.5),
            margin=2,
            value_at_margin=0,
            sigmoid="linear",
        ).mean()
        dont_rotate = (1 + dont_rotate) / 2.0
        alignment_reward = dont_rotate * pos_orient_reward * angle_alignment

        if self.move_speed == 0:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.5).mean()
            return dont_move * alignment_reward
        else:
            vel = center_of_mass_velocity[[0, 1]]
            com_velocity = np.linalg.norm(vel)
            move = rewards.tolerance(
                com_velocity,
                bounds=(
                    self.move_speed - 0.1 * self.move_speed,
                    self.move_speed + 0.1 * self.move_speed,
                ),
                margin=self.move_speed / 2,
                value_at_margin=0.5,
                sigmoid="gaussian",
            )
            move = (5 * move + 1) / 6
            if np.isclose(com_velocity, 0.0):
                angle_reward = 1.0
            else:
                direction = vel / (com_velocity + 1e-6)
                target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
                dot = target_direction.dot(direction)
                angle_reward = (dot + 1.0) / 2.0
            return alignment_reward * move * angle_reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^crawl-(\d+\.*\d*)-(\d+\.*\d*)-(u|d)$"
        match = re.search(pattern, name)
        if match:
            if match.group(3) == "d":
                direction = -1
            elif match.group(3) == "u":
                direction = 1
            spine_height, move_speed = float(match.group(1)), float(match.group(2))
            return CrawlReward(spine_height=spine_height, move_speed=move_speed, direction=direction)
        return None


@dataclasses.dataclass
class MoveAndRaiseArmsReward(RewardFunction):

    low_height: float = 0.6
    stand_height: float = 1.4
    move_speed: float = 5
    move_angle: float = 0
    stay_low: bool = False
    egocentric_target: bool = True
    visualize_target_angle: bool = True
    left_pose: str = "h"
    right_pose: str = "h"
    arm_coeff: float = 1.0
    loc_coeff: float = 1.0

    def __post_init__(self):
        self.locomotion_reward: LocomotionReward = LocomotionReward(
            move_speed=self.move_speed,
            stand_height=self.stand_height,
            move_angle=self.move_angle,
            egocentric_target=self.egocentric_target,
            low_height=self.low_height,
            stay_low=self.stay_low,
        )
        self.arms_reward: ArmsReward = ArmsReward(
            stand_height=self.stand_height,
            left_pose=self.left_pose,
            right_pose=self.right_pose,
        )
        loc_coeff = 1
        if self.left_pose == "l" or self.right_pose == "l":
            if self.left_pose == "m" or self.right_pose == "m":
                loc_coeff = 0.9
            elif self.left_pose == "h" or self.right_pose == "h":
                loc_coeff = 0.6
        elif self.left_pose == "m" or self.right_pose == "m":
            if self.left_pose == "m" or self.right_pose == "m":
                loc_coeff = 0.9
            elif self.left_pose == "h" or self.right_pose == "h":
                loc_coeff = 0.3
        elif self.left_pose == "h" and self.right_pose == "h":
            loc_coeff = 0.7
        self.loc_coeff = loc_coeff

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        locomotion_component = self.locomotion_reward.compute(
            model,
            data,
        )
        arms_component = self.arms_reward.compute(model, data)
        return (self.arm_coeff * arms_component + self.loc_coeff * locomotion_component) / (self.arm_coeff + self.loc_coeff)

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)-raisearms-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed = float(match.group(1)), float(match.group(2))
            left_pose, right_pose = match.group(3), match.group(4)
            return MoveAndRaiseArmsReward(
                move_angle=move_angle,
                move_speed=move_speed,
                left_pose=left_pose,
                right_pose=right_pose,
            )
        return None
