# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import json
from pathlib import Path
import imageio
import copy
import math
import numpy as np
import humenv
from humenv.misc.motionlib import MotionBuffer


Deg2Rad = math.pi / 180.0
ROTATION_AXIS_Z = [0, 1, 0]
ROTATION_AXIS_Y = [0, 0, 1]
ROTATION_AXIS_X = [1, 0, 0]


SAVE_MOTIONS = True
OUT_DIR = Path("goal_poses")
MOTION_FOLDER = Path("humenv_amass")
ZERO_VELOCITY = True
if ZERO_VELOCITY:
    JSON_FILE = "goals.json"
    JSON_FILE_OBS_ONLY = "goals_obs_only.json"
else:
    JSON_FILE = "goal_with_vel.json"
    JSON_FILE_OBS_ONLY = "goal_with_vel_obs_only.json"
TRAIN_DATASET = "test_train_split/large1_small1_train_0.1.txt"

MOTIONS = {
    # STAND (both legs on ground)
    "t_pose": {"motion": "0-MPI_HDM05_dg_HDM_dg_02-01_02_120_poses", "frame": 0},
    "t_pose_lower_arms": {"motion": "0-MPI_mosh_50020_simple_crouch_3_poses", "frame": 0, "rotate_deg": 180},
    "t_pose_bow_head": {"motion": "0-MPI_Limits_03100_rom4_poses", "frame": 70},
    "u_stretch_y_right": {"motion": "0-MPI_mosh_50002_stretches_poses", "frame": 170, "rotate_deg": 180},
    "u_stretch_y_left": {"motion": "0-MPI_mosh_50002_stretches_poses", "frame": 120, "rotate_deg": 180},
    "u_stretch_z_right": {"motion": "0-MPI_mosh_50002_stretches_poses", "frame": 270, "rotate_deg": 180},
    "u_stretch_z_left": {"motion": "0-MPI_mosh_50002_stretches_poses", "frame": 330, "rotate_deg": 180},
    "u_stretch_x_back": {"motion": "0-MPI_mosh_50002_stretches_poses", "frame": 15 * 30 + 10, "rotate_deg": 180},
    "u_stretch_x_front_part": {"motion": "0-MPI_mosh_50002_stretches_poses", "frame": 16 * 30 + 7, "rotate_deg": 180},
    "u_stretch_x_front_full": {"motion": "0-MPI_mosh_50002_stretches_poses", "frame": 17 * 30 + 15, "rotate_deg": 180},
    "crossed_arms": {"motion": "0-BMLmovi_Subject_13_F_MoSh_Subject_13_F_14_poses", "frame": 60},
    "scratching_head": {"motion": "0-BMLmovi_Subject_11_F_MoSh_Subject_11_F_14_poses", "frame": 60},
    "right_hand_wave": {"motion": "0-HumanEva_S1_Gestures_1_poses", "frame": 6 * 30, "rotate_deg": -90},
    "x_strech": {"motion": "0-BMLmovi_Subject_13_F_MoSh_Subject_13_F_15_poses", "frame": 2 * 30},
    "i_strecth": {"motion": "0-BMLmovi_Subject_35_F_MoSh_Subject_35_F_13_poses", "frame": 2 * 30},
    "arms_stretch": {"motion": "0-CMU_49_49_09_poses", "frame": 8 * 30, "rotate_deg": -90},
    "drinking_from_bottle": {"motion": "0-MPI_HDM05_mm_HDM_mm_08-01_01_120_poses", "frame": 9 * 30},
    "arm_on_chest": {"motion": "0-Eyes_Japan_Dataset_hamada_greeting-05-salute(chest)-hamada_poses", "frame": 17 * 30},
    "pre_throw": {"motion": "0-BioMotionLab_NTroje_rub046_0024_throwing_hard3_poses", "frame": 30 + 7, "rotate_deg": 90},
    "egyptian": {"motion": "0-KIT_291_walk_like_an_egyptian05_poses", "frame": 2 * 30 + 15},
    "zombie": {"motion": "0-Eyes_Japan_Dataset_kaiwa_pose-20-zombee-kaiwa_poses", "frame": 4 * 30 - 2, "rotate_deg": 180},
    "stand_martial_arts": {"motion": "0-ACCAD_Male2MartialArtsKicks_c3d_G11-roundhouseleadingright_poses", "frame": 98, "rotate_deg": 110},
    "peekaboo": {"motion": "0-CMU_113_113_12_poses", "frame": 30},
    "dance": {"motion": "0-KIT_572_dance_chacha02_poses", "frame": 60, "rotate_deg": 90},
    # CROUCH
    "kneel_left": {"motion": "0-ACCAD_Male2General_c3d_A7-Crouch_poses", "frame": 130, "rotate_deg": 180},
    "crouch_high": {"motion": "0-CMU_136_136_13_poses", "frame": 30},
    "crouch_medium": {"motion": "0-CMU_136_136_09_poses", "frame": 2 * 30, "rotate_deg": 180},
    "crouch_low": {"motion": "0-ACCAD_Male1General_c3d_GeneralA8-CrouchtoLieDown_poses", "frame": 30, "rotate_deg": 180},
    # SQUAT
    "squat_pre_jump": {"motion": "0-CMU_118_118_05_poses", "frame": 60 - 5, "rotate_deg": -90},
    "squat_hands_on_ground": {"motion": "0-CMU_111_111_08_poses", "frame": 300 - 8, "rotate_deg": -60},
    # SINGLE-LEG STANDING
    "side_high_kick": {"motion": "0-ACCAD_Male2MartialArtsKicks_c3d_G3-Sidekickleadingright_poses", "frame": 25, "rotate_deg": 180},
    "pre_front_kick": {"motion": "0-ACCAD_Male2MartialArtsKicks_c3d_G3-frontkick_poses", "frame": 20, "rotate_deg": 120},
    "arabesque_hold_foot": {"motion": "0-BMLmovi_Subject_34_F_MoSh_Subject_34_F_8_poses", "frame": 60, "rotate_deg": -90},
    "hold_right_foot": {"motion": "0-BMLmovi_Subject_40_F_MoSh_Subject_40_F_11_poses", "frame": 5 * 30 + 15},
    "hold_left_foot": {"motion": "0-BMLmovi_Subject_86_F_MoSh_Subject_86_F_20_poses", "frame": 4 * 30},
    "bend_on_left_leg": {"motion": "0-BioMotionLab_NTroje_rub034_0030_scamper_rom_poses", "frame": 5 * 30, "rotate_deg": 270},
    # LIE
    "lie_front": {"motion": "0-ACCAD_Male2General_c3d_A9-Lie_poses", "frame": 30, "rotate_deg": 90},
    "crawl_backward": {"motion": "0-ACCAD_Male1General_c3d_GeneralA12-MilitaryCrawlBackwards_poses", "frame": 4 * 30, "rotate_deg": -90},
    "lie_back_knee_bent": {"motion": "0-CMU_111_111_08_poses", "frame": 15, "rotate_deg": 45},
    "lie_side": {"motion": "0-CMU_111_111_08_poses", "frame": 3 * 30, "rotate_deg": -45},
    "crunch": {"motion": "0-MPI_HDM05_dg_HDM_dg_03-09_01_120_poses", "frame": 7 * 30},
    "lie_back": {"motion": "0-ACCAD_Male1General_c3d_GeneralA8-CrouchtoLieDown_poses", "frame": 5 * 30, "rotate_deg": 180},
    # SIT
    "sit_side": {"motion": "0-CMU_111_111_08_poses", "frame": 5 * 30 + 5, "rotate_deg": -45},
    "sit_hand_on_legs": {"motion": "0-CMU_82_82_05_poses", "frame": 2, "rotate_deg": 180},
    "sit_hand_behind": {"motion": "0-CMU_82_82_05_poses", "frame": 2 * 30, "rotate_deg": 180},
    # FOUR POINTS OF CONACT
    "knees_and_hands": {"motion": "0-CMU_111_111_08_poses", "frame": 8 * 30 - 7, "rotate_deg": -80},
    "bridge_front": {"motion": "0-MPI_HDM05_dg_HDM_dg_03-09_01_120_poses", "frame": 30 * 25, "rotate_deg": 180},
    "push_up": {"motion": "0-MPI_HDM05_dg_HDM_dg_03-09_01_120_poses", "frame": 30 * 26 + 7, "rotate_deg": 180},
    # ACROBATIC
    "handstand": {"motion": "0-CMU_85_85_05_poses", "frame": 6 * 30, "rotate_deg": 180},
    "handstand_right_leg_bent": {"motion": "0-CMU_85_85_13_poses", "frame": 5 * 30 + 15, "rotate_deg": 90},
}


def generate_poses_datasets(out_dir):
    kwargs = {
        "render_width": 1024,
        "render_height": 768,
    }
    env, _ = humenv.make_humenv(**kwargs)
    out_dir.mkdir(exist_ok=True, parents=True)
    json_contents = {}
    json_contents_obs_only = {}
    id = 0
    in_train_motions_cnt = 0
    with open(TRAIN_DATASET, "r") as file:
        train_motions = file.readlines()
    train_motions = [x.strip().replace(".hdf5", "") for x in train_motions]
    for motion_name, value in MOTIONS.items():
        motion_dataset = "Test"
        if value["motion"] in train_motions:
            in_train_motions_cnt += 1
            motion_dataset = "Train"
        motion_buffer = MotionBuffer([f"{value['motion']}.hdf5"], base_path=str(MOTION_FOLDER))
        motion = motion_buffer.storage[0]
        idx = value["frame"]
        qpos = copy.deepcopy(motion["qpos"][idx])
        if "rotate_deg" in value:
            orientation = create_angle_axis_quaternion(np.array(ROTATION_AXIS_Y), value["rotate_deg"])
            update_qpos_heading(qpos, orientation)
        qvel = np.zeros_like(motion["qvel"][idx]) if ZERO_VELOCITY else motion["qvel"][idx]
        env.unwrapped.set_physics(qpos=qpos, qvel=qvel)
        observation = env.unwrapped.get_obs()["proprio"] if ZERO_VELOCITY else motion["observation"][idx]
        frames = [env.unwrapped.render()]
        out_file = out_dir / f"{motion_name}.jpg"
        print(f"Motion: {motion_name} from dataset: {motion_dataset}")
        if SAVE_MOTIONS:
            print(f"Saving {out_file}")
            imageio.mimsave(out_file, frames)
            json_contents[motion_name] = {
                "qpos": qpos.tolist(),
                "qvel": qvel.tolist(),
                "observation": observation.tolist(),
                "id": id,
            }
            json_contents_obs_only[motion_name] = observation.tolist()
        id += 1
    env.close()
    if SAVE_MOTIONS:
        with open(out_dir / JSON_FILE, mode="w") as poses_file:
            json.dump(json_contents, poses_file)
        with open(out_dir / JSON_FILE_OBS_ONLY, mode="w") as poses_file:
            json.dump(json_contents_obs_only, poses_file)
    print(f"Processed {len(MOTIONS)} motions")
    print(f"No. of motions from Train: {in_train_motions_cnt}")
    print(f"No. of motions from Test: {len(MOTIONS) - in_train_motions_cnt}")


def update_qpos_heading(qpos, orientation):
    heading = qpos[3:7]
    heading = mult_quaternion(orientation, heading)
    for reset_idx in range(3, 7):
        qpos[reset_idx] = heading[reset_idx - 3]


def create_angle_axis_quaternion(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    radians = angle_degrees * Deg2Rad
    halfAngle = radians * 0.5
    s = math.sin(halfAngle)
    c = math.cos(halfAngle)

    x = axis[0] * s
    y = axis[1] * s
    z = axis[2] * s
    w = c

    return np.array([w, x, y, z])


def mult_quaternion(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    q1w = lhs[0]
    q1x = lhs[1]
    q1y = lhs[2]
    q1z = lhs[3]

    q2w = rhs[0]
    q2x = rhs[1]
    q2y = rhs[2]
    q2z = rhs[3]

    cx = q1y * q2z - q1z * q2y
    cy = q1z * q2x - q1x * q2z
    cz = q1x * q2y - q1y * q2x

    dot = q1x * q2x + q1y * q2y + q1z * q2z

    x = q1x * q2w + q2x * q1w + cx
    y = q1y * q2w + q2y * q1w + cy
    z = q1z * q2w + q2z * q1w + cz
    w = q1w * q2w - dot

    return np.array([w, x, y, z])


if __name__ == "__main__":
    generate_poses_datasets(out_dir=OUT_DIR)
