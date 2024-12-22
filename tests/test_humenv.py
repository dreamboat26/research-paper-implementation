# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
from gymnasium.wrappers import FlattenObservation
from humenv.env import HumEnv
from humenv.rewards import HeadstandReward, MoveAndRaiseArmsReward, RewardFunction


@pytest.mark.parametrize("reward", [HeadstandReward, MoveAndRaiseArmsReward])
@pytest.mark.parametrize("state_init", ["Default", "Fall", "DefaultAndFall"])
def test_initalization(state_init: str, reward: RewardFunction, horizon: int = 30, seed: int = 10):
    env = HumEnv(task=reward(), state_init=state_init)
    env = FlattenObservation(env)
    env.reset(seed=seed)
    for tt in range(horizon):
        action = (np.random.rand(69) * 2) - 1
        obs, rewards, terminated, truncated, info = env.step(action)
    assert env is not None
    assert obs is not None
