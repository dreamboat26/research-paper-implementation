"""
Create HDF5 files containing qpos/qvel info from AMASS

Copyright (c) 2023 Carnegie Mellon University
Copyright (c) 2018-2023, NVIDIA Corporation

Adapted from:
https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/process_amass_raw.py
https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/process_amass_db.py
"""

import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "SMPLSim")
sys.path.insert(0, "PHC/scripts/data_process")
sys.path.insert(0, "..")

from pathlib import Path
import numpy as np
import joblib
import tqdm
import json
import torch
from easydict import EasyDict
from smpl_sim.smpllib.motion_lib_smpl import MotionLibSMPL, FixHeightMode

# from SMPLSim.smpl_utils.motion_lib_smpl import MotionLibSMPL, FixHeightMode
# from phc.utils.motion_lib_smpl import MotionLibSMPL , FixHeightMode
import multiprocessing
import mujoco
import typing as tp
import time
from collections import defaultdict
import functools
from rich import print
import h5py

PHC_FILES_TO_REMOVE = "PHC/sample_data/amass_copycat_occlusion_v3.pkl"
SMPL_FOLDER = "AMASS/models"
AMASS_ROOT_FOLDER = "AMASS/datasets"
OUTPUT_DIR = "humenv_amass"

all_sequences = [
    "ACCAD",
    "BMLhandball",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "CMU",
    "DFaust_67",
    "DanceDB",
    "EKUT",
    "Eyes_Japan_Dataset",
    "MPI_HDM05",
    "HumanEva",
    "KIT",
    "MPI_mosh",
    "MPI_Limits",
    "SFU",
    "SSM_synced",
    "TCD_handMocap",
    "TotalCapture",
    "Transitions_mocap",
]


def save_hdf5(
    file_path: str | Path,
    episode: tp.Dict[str, np.ndarray],
    include_only: tp.List | None = None,
) -> None:
    file_path = Path(file_path)
    assert file_path.suffix == ".hdf5"
    keys_to_use = list(episode.keys())
    if include_only is not None and len(include_only) > 0:
        keys_to_use = [k for k in keys_to_use if k in include_only]

    # for k,v in episode.items():
    #     print(k, v.shape, v.dtype)

    hf = h5py.File(file_path, "w")
    hf.attrs["num_episodes"] = 1
    grp = hf.create_group("ep_0")
    grp.attrs["length"] = episode["qpos"].shape[0]
    for k in keys_to_use:
        grp.create_dataset(k, data=episode[k], compression="gzip")
    hf.close()


def replay_and_save(
    motion_ids: tp.List[tp.Any],
    config: EasyDict,
    output_dir: str,
    semaphore: tp.Any,
    motion_name_mapper: tp.Dict,
):
    from smpl_sim.smpllib.torch_smpl_humanoid_batch import (
        Humanoid_Batch,
    )  # from SMPLSim
    from smpl_sim.smpllib.smpl_parser import SMPL_Parser  # from SMPLSim
    from humenv.env import HumEnv

    smpl_type = config.smpl_type
    data_dir = config.data_dir
    assert smpl_type == "smpl"
    smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
    smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
    smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")
    humanoid = Humanoid_Batch(data_dir=data_dir, filter_vel=False)
    mesh_parsers = EasyDict(
        {"0": smpl_parser_n, "1": smpl_parser_m, "2": smpl_parser_f, "batch": humanoid}
    )
    with semaphore:
        print(
            f"{multiprocessing.current_process().name} needs to parse {len(motion_ids)} motions"
        )

        start_t = time.time()
        data = joblib.load(config.motion_file)
        values = {k: data[k] for k in motion_ids}
        del data
        print(
            f"{multiprocessing.current_process().name} has loaded the data in {time.time()-start_t:.2f} s"
        )

    FPS: int = 30
    env = HumEnv()
    env_dt = env.model.opt.timestep * env.action_repeat

    try:
        disable = multiprocessing.current_process()._identity[0] != 2
    except:
        disable = False

    # new_episodes = defaultdict(list)
    total_len = 0
    output_dir = Path(output_dir)
    for m_id in tqdm.tqdm(motion_ids, disable=disable):
        res = MotionLibSMPL.load_motion_with_skeleton(
            np.array([0]),
            [values[str(m_id)]],
            [np.zeros(17)],
            mesh_parsers,
            config,
            None,
            0,
        )
        curr_motion = res[0][1]
        motion_fps = curr_motion.fps
        observation, _ = env.reset()
        num_frames = int(curr_motion.global_translation.shape[0])
        motion_steps = int(num_frames * FPS / motion_fps)
        assert motion_fps == 30
        assert num_frames == curr_motion.qpos.shape[0]
        assert motion_steps == num_frames
        total_len += motion_steps
        episode = defaultdict(list)
        # check computation with mujoco
        rrr = []
        for cur_t in range(0, motion_steps - 1):
            qpos2 = curr_motion.qpos[cur_t + 1].flatten().numpy()
            qpos1 = curr_motion.qpos[cur_t].flatten().numpy()
            new_qvel = np.zeros(env.model.nv)
            assert np.isclose(env_dt, 1 / 30.0)
            mujoco.mj_differentiatePos(env.model, new_qvel, env_dt, qpos1, qpos2)
            erro = np.abs(new_qvel - curr_motion.qvel[cur_t].flatten().numpy())
            rrr.append(erro)
        rrr = np.array(rrr)
        eo = np.mean(rrr, axis=0)
        assert np.all(eo[:3] < 1e-5), eo[:3]
        assert np.all(eo[6:] < 1e-5), eo[6:]
        # breakpoint()

        for cur_t in range(motion_steps):
            qpos = curr_motion.qpos[cur_t].flatten().numpy()
            qvel = curr_motion.qvel[cur_t].flatten().numpy()
            env.set_physics(qpos=qpos, qvel=qvel)
            observation = env.get_obs()["proprio"]
            episode["observation"].append(observation)
            episode["qpos"].append(qpos)
            episode["qvel"].append(qvel)
            episode["motion_id"].append(motion_name_mapper["name2id"][m_id])
            episode["truncated"].append(True if cur_t == motion_steps - 1 else False)
            episode["terminated"].append(False)

        for k, v in episode.items():
            v = np.array(v)
            if len(v.shape) == 1:
                v = v.reshape(-1, 1)
            episode[k] = v

        save_hdf5(output_dir / f'{str(m_id).replace(" ", "")}.hdf5', episode)


def _hdf5_step(motion_file, motion_name_mapper, num_workers=0, output_dir=OUTPUT_DIR):
    Path(output_dir).mkdir(exist_ok=True)
    randomize_heading = False
    motion_lib_cfg = EasyDict(
        {
            "motion_file": motion_file,
            "device": torch.device("cpu"),
            "fix_height": FixHeightMode.full_fix,
            "min_length": -1,
            "max_length": -1,
            # "multi_thread": True,
            "smpl_type": "smpl",
            "randomrize_heading": randomize_heading,
            "data_dir": "data_preparation/AMASS/models",
        }
    )
    data = joblib.load(motion_lib_cfg.motion_file)
    motion_keys = np.array(list(data.keys()))
    del data

    with open(motion_name_mapper, "r") as f:
        motion_name_mapper = json.load(f)

    print(f"Num motions to process: {motion_keys.shape}")
    m = multiprocessing.Manager()
    semaphore = m.Semaphore(10)

    print(
        f"using {num_workers} workers out of {multiprocessing.cpu_count()}. (see python preocess_amass.py -h for info)"
    )
    if num_workers == 0:
        results = replay_and_save(
            motion_keys,
            config=motion_lib_cfg,
            output_dir=output_dir,
            semaphore=semaphore,
            motion_name_mapper=motion_name_mapper,
        )
    else:
        ctx = multiprocessing.get_context("spawn")
        list_eps = np.array_split(motion_keys, num_workers)
        assert len(list_eps) == num_workers
        with ctx.Pool(num_workers) as pool:
            f = functools.partial(
                replay_and_save,
                config=motion_lib_cfg,
                output_dir=output_dir,
                semaphore=semaphore,
                motion_name_mapper=motion_name_mapper,
            )
            list_res = pool.map(f, list_eps)


def _filter(folder, phc_files_to_remove):
    from process_amass_db import process_qpos_list  # from PHC
    from smpl_sim.smpllib.smpl_parser import SMPL_Parser  # from SMPLSim

    root = Path(folder)
    print(f"Processing AMASS files from {root}")
    motions = {}
    for dataset in all_sequences:
        print(dataset)
        if (root / dataset).is_dir():
            for file in tqdm.tqdm((root / dataset).glob("**/*.npz")):
                if not str(file).endswith("shape.npz"):
                    name = f"{dataset}_{file.parent.stem}_{file.name[:-4]}"
                    motions[name] = dict(np.load(str(file)))
        else:
            print("[WARNING] dataset not found. Skipping it.")

    smpl_parser_n = SMPL_Parser(
        model_path=SMPL_FOLDER, gender="neutral", use_pca=False, create_transl=False
    )
    smpl_parser_m = SMPL_Parser(
        model_path=SMPL_FOLDER, gender="male", use_pca=False, create_transl=False
    )
    smpl_parser_f = SMPL_Parser(
        model_path=SMPL_FOLDER, gender="female", use_pca=False, create_transl=False
    )

    motion_list = list(motions.items())
    phc_files_to_remove = joblib.load(phc_files_to_remove)
    amass_seq_data = process_qpos_list(
        motion_list,
        target_fr=30,
        amass_occlusion=phc_files_to_remove,
        smpl_parser_n=smpl_parser_n,
        smpl_parser_m=smpl_parser_m,
        smpl_parser_f=smpl_parser_f,
    )
    motion_name_mapped = {"name2id": {}, "id2name": {}}
    for _num, _name in enumerate(sorted(amass_seq_data.keys())):
        motion_name_mapped["name2id"][_name] = _num
        motion_name_mapped["id2name"][_num] = _name
    return amass_seq_data, motion_name_mapped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AMASS dataset processing.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=AMASS_ROOT_FOLDER,
        help="Path to the AMASS data directory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of parallel workers",
    )
    args = parser.parse_args()
    if not Path("amass_seq_data.pkl").exists():
        amass_seq_data, motion_name_mapped = _filter(
            args.data_path, phc_files_to_remove=PHC_FILES_TO_REMOVE
        )
        joblib.dump(amass_seq_data, "amass_seq_data.pkl")
        with open("motion_name_mapper.json", "w") as f:
            json.dump(motion_name_mapped, f)
    else:
        print("Found 'amass_seq_data.pkl', loading it...", end=" ", flush=True)
        amass_seq_data = joblib.load("amass_seq_data.pkl")
        print("done")
    _hdf5_step(
        "amass_seq_data.pkl",
        "motion_name_mapper.json",
        num_workers=args.num_workers,
        output_dir=OUTPUT_DIR,
    )
