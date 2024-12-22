# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import gymnasium
from humenv.rewards import HeadstandReward, MoveAndRaiseArmsReward
from humenv import ALL_TASKS


@pytest.mark.parametrize("reward", [HeadstandReward, MoveAndRaiseArmsReward])
def test_register_reward_object(reward: str, num_envs: int = 3):
    envs = gymnasium.make_vec("humenv/HumEnv-v0.0.1", task=reward(), num_envs=num_envs)
    observations, infos = envs.reset(seed=1)
    actions = envs.action_space.sample()
    observations, rewards, terminations, truncations, infos = envs.step(actions)
    assert observations
    assert infos


@pytest.mark.parametrize("reward", ALL_TASKS)
def test_register_reward_str(reward: str, num_envs: int = 3):
    envs = gymnasium.make_vec("humenv/HumEnv-v0.0.1", task=reward, num_envs=num_envs)
    observations, infos = envs.reset(seed=1)
    actions = envs.action_space.sample()
    observations, rewards, terminations, truncations, infos = envs.step(actions)
    assert observations
    assert infos
