# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

#set -x # Print trace of commands
set -e # Exit immediately on failure
set -u

MODEL_PATH="models"

process_data() {
    need_cmd mkdir
    need_cmd git
    need_cmd conda
    need_cmd bash

    remove_chumpy_dependence

    conda deactivate

    git clone https://github.com/ZhengyiLuo/PHC.git
    git clone https://github.com/ZhengyiLuo/SMPLSim.git
    
    conda create -y -n temp python=3.10
    conda activate temp
    pip3 install torch --index-url https://download.pytorch.org/whl/cpu
    pip3 install rich joblib gdown tqdm "gymnasium>=0.26" "mujoco==3.2.3" "dm_control==1.0.23" "git+https://github.com/ZhengyiLuo/smplx.git@master" easydict numpy-stl torchgeometry opencv-python imageio pyyaml h5py hydra-core numpy

    
    cd PHC
    git checkout 34fa3a1c42c519895bc33ae47a10a1ef61a39520
    git apply ../phc_patch.patch 
    bash download_data.sh
    cd ..

    cd SMPLSim
    git checkout 3bcc506d92bf15329b2d68efcf429725b67f3a06
    git apply ../smplsim_patch.patch 
    cd ..

    python process_amass.py --num_workers 0
}

remove_chumpy_dependence() {
    need_cmd mkdir
    need_cmd git
    need_cmd conda

    conda deactivate

    DATA_PATH=$(pwd)

    git clone https://github.com/vchoutas/smplx.git
    cd smplx
    conda create -y -n py27 python=2.7
    conda activate py27
    pip install tqdm chumpy
    mkdir ${DATA_PATH}/AMASS/models
    python tools/clean_ch.py --input-models  ${DATA_PATH}/AMASS/models_with_chumpy/*.pkl --output-folder ${DATA_PATH}/AMASS/models
    cd ${DATA_PATH}
    rm -rf smplx
    conda deactivate
    conda env remove -y --name py27
}

generate_goal_poses() {
    need_cmd git
    need_cmd conda
    need_cmd bash

    conda deactivate

    conda create -y -n temp_humenv python=3.10
    conda activate temp_humenv

    pip3 install "git+https://github.com/facebookresearch/humenv.git@main" imageio
    python goal_pose_selection.py
    conda deactivate
    conda env remove -y --name temp_humenv
}

need_cmd() {
    if ! check_cmd "$1"
    then err "need '$1' (command not found)"
    fi
}

check_cmd() {
    command -v "$1" > /dev/null 2>&1
    return $?
}

process_data "$@" || exit 1
generate_goal_poses || exit 1
