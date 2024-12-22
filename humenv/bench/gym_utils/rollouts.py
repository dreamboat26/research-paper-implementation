# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import gymnasium
from gymnasium.vector.vector_env import VectorEnv
import torch
from typing import Any, Dict, List, Tuple
from .episodes import Episode, CollectEpisodes
from packaging.version import Version


if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):
    from gymnasium.wrappers import AutoResetWrapper as _autoresetwrapper
elif Version(gymnasium.__version__) >= Version("1.0"):
    from gymnasium.wrappers import Autoreset as _autoresetwrapper
else:
    raise ValueError(f"unsupported gymnasium version {gymnasium.__version__}")


def rollout(env: Any, agent: Any, num_episodes: int, ctx: torch.Tensor | None = None) -> Tuple[Dict[str, Any], List[Episode]]:
    if isinstance(env, VectorEnv):
        return _parallel_env_rollout(env=env, agent=agent, num_episodes=num_episodes, ctx=ctx)
    else:
        if isinstance(env, _autoresetwrapper):
            raise ValueError("We don't support the autoreset wrapper yet in the rollout function")
        return _single_env_rollout(env=env, agent=agent, num_episodes=num_episodes, ctx=ctx)


def _single_env_rollout(
    env: Any, agent: Any, num_episodes: int, ctx: torch.Tensor | None = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    returns, lengths, episodes = [], [], []
    curr_return, curr_length, _stop = 0.0, 0, False
    observation, info = env.reset()
    _episode = Episode()
    _episode.initialise(observation, info)
    ctx = {} if ctx is None else {"z": ctx}
    while not _stop:
        input_dict = {"obs": observation}
        input_dict.update(ctx)
        action = agent.act(**input_dict)
        observation, reward, terminated, truncated, info = env.step(action)
        _episode.add(observation, reward, action, terminated, truncated, info)
        done = terminated or truncated
        curr_return += reward
        curr_length += 1
        if done:
            episodes.append(_episode.get())
            returns.append(curr_return)
            lengths.append(curr_length)
            curr_return, curr_length = 0.0, 0
            observation, info = env.reset()
            _episode = Episode()
            _episode.initialise(observation, info)
            if len(returns) >= num_episodes:
                _stop = True
    return {"reward": returns, "length": lengths}, episodes


def _parallel_env_rollout(
    env: Any, agent: Any, num_episodes: int, ctx: torch.Tensor | None = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    _stop = False
    episodes = CollectEpisodes(batch_size=env.num_envs)
    observation, info = env.reset()
    episodes.initialise(observation, info)
    ctx = {} if ctx is None else {"z": ctx}
    while not _stop:
        input_dict = {"obs": observation}
        input_dict.update(ctx)
        action = agent.act(**input_dict)
        observation, reward, terminated, truncated, info = env.step(action)
        episodes.add(observation, reward, action, terminated, truncated, info)
        if len(episodes) >= num_episodes:
            _stop = True
    return {"reward": episodes.returns, "length": episodes.lengths}, episodes.get()
