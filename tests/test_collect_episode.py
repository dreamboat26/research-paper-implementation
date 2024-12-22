# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import gymnasium
from gymnasium import spaces
import numpy as np
from humenv.bench.gym_utils.rollouts import rollout
import pytest


class StepEnv(gymnasium.Env):
    def __init__(self, goal: int = 10, horizon: int = 10, init_state: int = 0):
        self.goal = goal
        self.horizon = horizon
        self.init_state = init_state

        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=float),
            }
        )
        self.action_space = spaces.Box(-1_000, 1_000, shape=(1,), dtype=float)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        self.observation = np.array([self.init_state], dtype=np.float32)
        info = {"square": self.observation**2}
        return {"proprio": np.array(self.observation)}, info

    def step(self, action):
        self.observation += action
        self.time_step += 1
        terminated = np.isclose(self.observation, self.goal).item()
        reward = 1 if terminated else 0
        truncated = self.time_step >= self.horizon
        info = {"square": self.observation**2}
        return (
            {"proprio": np.array(self.observation)},
            reward,
            terminated,
            truncated,
            info,
        )


class FakeAgent:
    def __init__(self, num_env=1):
        self.num_env = num_env

    def act(self, **kwargs):
        if self.num_env == 1:
            return np.ones(1)
        else:
            return np.ones((self.num_env, 1))


def test_single_env():
    num_episodes = 2
    env = StepEnv(goal=10, horizon=3)
    agent = FakeAgent()

    stats, episodes = rollout(env, agent, num_episodes=num_episodes)

    assert len(episodes) == num_episodes
    for ep in episodes:
        assert np.allclose(ep["observation"]["proprio"], np.array([0, 1, 2, 3]).reshape(-1, 1))
        assert np.allclose(ep["action"], np.ones(3).reshape(-1, 1))
        assert np.allclose(ep["reward"], np.zeros(3, dtype=bool))
        assert np.allclose(ep["terminated"], np.zeros(3, dtype=bool))
        assert np.allclose(ep["truncated"], np.array([0, 0, 1], dtype=bool))
        assert np.allclose(ep["info"]["square"], np.array([0, 1, 2, 3]).reshape(-1, 1) ** 2)


@pytest.mark.parametrize("parallel_class", [gymnasium.vector.SyncVectorEnv])
def test_parallel_env(parallel_class):
    num_episodes = 3
    env = parallel_class(
        [
            lambda: StepEnv(goal=10, horizon=3, init_state=1),
            lambda: StepEnv(goal=10, horizon=3),
            lambda: StepEnv(goal=2, horizon=3),
        ]
    )
    agent = FakeAgent(num_env=env.num_envs)

    stats, episodes = rollout(env, agent, num_episodes=num_episodes)

    assert len(episodes) == num_episodes
    ep = episodes[1]
    assert np.allclose(ep["observation"]["proprio"], np.array([1, 2, 3, 4]).reshape(-1, 1))
    assert np.allclose(ep["action"], np.ones(3).reshape(-1, 1))
    assert np.allclose(ep["reward"], np.array([0, 0, 0]))
    assert np.allclose(ep["terminated"], np.zeros(3, dtype=bool))
    assert np.allclose(ep["truncated"], np.array([0, 0, 1], dtype=bool))
    assert np.allclose(ep["info"]["square"], np.array([1, 2, 3, 4]).reshape(-1, 1) ** 2)
    ep = episodes[2]
    assert np.allclose(ep["observation"]["proprio"], np.array([0, 1, 2, 3]).reshape(-1, 1))
    assert np.allclose(ep["action"], np.ones(3).reshape(-1, 1))
    assert np.allclose(ep["reward"], np.array([0, 0, 0]))
    assert np.allclose(ep["terminated"], np.array([0, 0, 0], dtype=bool))
    assert np.allclose(ep["truncated"], np.array([0, 0, 1], dtype=bool))
    assert np.allclose(ep["info"]["square"], np.array([0, 1, 2, 3]).reshape(-1, 1) ** 2)
    ep = episodes[0]
    assert np.allclose(ep["observation"]["proprio"], np.array([0, 1, 2]).reshape(-1, 1))
    assert np.allclose(ep["action"], np.ones(2).reshape(-1, 1))
    assert np.allclose(ep["reward"], np.array([0, 1]))
    assert np.allclose(ep["terminated"], np.array([0, 1], dtype=bool))
    assert np.allclose(ep["truncated"], np.array([0, 0], dtype=bool))
    assert np.allclose(ep["info"]["square"], np.array([0, 1, 2]).reshape(-1, 1) ** 2)


@pytest.mark.parametrize("parallel_class", [gymnasium.vector.SyncVectorEnv])
def test_parallel_env_2(parallel_class):
    num_episodes = 2
    env = parallel_class(
        [
            lambda: StepEnv(goal=10, horizon=3, init_state=1),
            lambda: StepEnv(goal=3, horizon=2),
        ]
    )
    agent = FakeAgent(num_env=env.num_envs)

    stats, episodes = rollout(env, agent, num_episodes=num_episodes)
    assert len(episodes) == num_episodes
    ep = episodes[0]
    assert np.allclose(ep["observation"]["proprio"], np.array([0, 1, 2]).reshape(-1, 1))
    assert np.allclose(ep["action"], np.ones(2).reshape(-1, 1))
    assert np.allclose(ep["reward"], np.array([0, 0]))
    assert np.allclose(ep["terminated"], np.zeros(2, dtype=bool))
    assert np.allclose(ep["truncated"], np.array([0, 1], dtype=bool))
    assert np.allclose(ep["info"]["square"], np.array([0, 1, 2]).reshape(-1, 1) ** 2)
    ep = episodes[1]
    assert np.allclose(ep["observation"]["proprio"], np.array([1, 2, 3, 4]).reshape(-1, 1))
    assert np.allclose(ep["action"], np.ones(3).reshape(-1, 1))
    assert np.allclose(ep["reward"], np.array([0, 0, 0]))
    assert np.allclose(ep["terminated"], np.zeros(3, dtype=bool))
    assert np.allclose(ep["truncated"], np.array([0, 0, 1], dtype=bool))
    assert np.allclose(ep["info"]["square"], np.array([1, 2, 3, 4]).reshape(-1, 1) ** 2)


@pytest.mark.parametrize("parallel_class", [gymnasium.vector.AsyncVectorEnv])
def test_parallel_async_env_2(parallel_class):
    num_episodes = 2
    env = parallel_class(
        [
            lambda: StepEnv(goal=10, horizon=3, init_state=1),
            lambda: StepEnv(goal=3, horizon=2),
        ]
    )
    agent = FakeAgent(num_env=env.num_envs)

    stats, episodes = rollout(env, agent, num_episodes=num_episodes)
    assert len(episodes) == num_episodes
    ep = episodes[0]
    assert np.allclose(ep["observation"]["proprio"], np.array([0, 1, 2]).reshape(-1, 1))
    assert np.allclose(ep["action"], np.ones(2).reshape(-1, 1))
    assert np.allclose(ep["reward"], np.array([0, 0]))
    assert np.allclose(ep["terminated"], np.zeros(2, dtype=bool))
    assert np.allclose(ep["truncated"], np.array([0, 1], dtype=bool))
    assert np.allclose(ep["info"]["square"], np.array([0, 1, 2]).reshape(-1, 1) ** 2)
    ep = episodes[1]
    assert np.allclose(ep["observation"]["proprio"], np.array([1, 2, 3, 4]).reshape(-1, 1))
    assert np.allclose(ep["action"], np.ones(3).reshape(-1, 1))
    assert np.allclose(ep["reward"], np.array([0, 0, 0]))
    assert np.allclose(ep["terminated"], np.zeros(3, dtype=bool))
    assert np.allclose(ep["truncated"], np.array([0, 0, 1], dtype=bool))
    assert np.allclose(ep["info"]["square"], np.array([1, 2, 3, 4]).reshape(-1, 1) ** 2)


if __name__ == "__main__":
    test_parallel_async_env_2(gymnasium.vector.AsyncVectorEnv)
