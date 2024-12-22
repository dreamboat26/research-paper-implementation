# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import gymnasium
from gymnasium.wrappers import FlattenObservation
from humenv.rewards import RewardFunction, HeadstandReward, MoveAndRaiseArmsReward
from humenv import ALL_TASKS


@pytest.mark.parametrize("reward", [HeadstandReward, MoveAndRaiseArmsReward])
def test_register_reward_object(reward: RewardFunction, seed: int = 1, horizon: int = 30):
    env = gymnasium.make("humenv/HumEnv-v0.0.1", task=reward())
    env = FlattenObservation(env)
    env.reset(seed=seed)
    for tt in range(horizon):
        action = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(action)
    assert env is not None
    assert obs is not None


@pytest.mark.parametrize("reward", ALL_TASKS)
def test_register_reward_string(reward: str, seed: int = 1, horizon: int = 30):
    env = gymnasium.make("humenv/HumEnv-v0.0.1", task=reward)
    env = FlattenObservation(env)
    env.reset(seed=seed)
    for tt in range(horizon):
        action = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(action)
    assert env is not None
    assert obs is not None
