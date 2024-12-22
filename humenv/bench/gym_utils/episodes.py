# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from collections.abc import Mapping
from collections import defaultdict
import dataclasses
from typing import Dict, List
import copy
import gymnasium
from packaging.version import Version


@dataclasses.dataclass
class Episode:
    storage: Dict | None = None

    def initialise(self, observation: np.ndarray | Dict[str, np.ndarray], info: Dict) -> None:
        if self.storage is None:
            self.storage = defaultdict(list)
        if isinstance(observation, Mapping):
            self.storage["observation"] = {k: [copy.deepcopy(v)] for k, v in observation.items()}
        else:
            self.storage["observation"].append(copy.deepcopy(observation))
        self.storage["info"] = {
            k: [copy.deepcopy(v)] for k, v in info.items() if not k.startswith("_") and k not in ["final_observation", "final_info"]
        }

    def add(
        self,
        observation: np.ndarray | Dict[str, np.ndarray],
        reward: np.ndarray,
        action: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: Dict,
    ) -> None:
        if isinstance(observation, Mapping):
            for k, v in observation.items():
                self.storage["observation"][k].append(copy.deepcopy(v))
        else:
            self.storage["observation"].append(copy.deepcopy(observation))
        for k, v in info.items():
            if not k.startswith("_") and k not in ["final_observation", "final_info"]:
                self.storage["info"][k].append(copy.deepcopy(v))
        self.storage["reward"].append(copy.deepcopy(reward))
        self.storage["action"].append(copy.deepcopy(action))
        self.storage["terminated"].append(copy.deepcopy(terminated))
        self.storage["truncated"].append(copy.deepcopy(truncated))

    def get(self) -> Dict[str, np.ndarray]:
        output = {}
        for k, v in self.storage.items():
            if k in ["observation", "info"]:
                if isinstance(v, Mapping):
                    output[k] = {}
                    for k2, v2 in v.items():
                        output[k][k2] = np.array(v2)
                else:
                    output[k] = np.array(v)
            else:
                output[k] = np.array(v)
        return output


if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):

    @dataclasses.dataclass
    class CollectEpisodes:
        batch_size: int
        completed_episodes: List[Episode] = dataclasses.field(default_factory=list)
        returns: List[float] = dataclasses.field(default_factory=list)
        lengths: List[int] = dataclasses.field(default_factory=list)
        ongoing_episodes: List[Episode] = dataclasses.field(default_factory=list)

        def initialise(self, observation: np.ndarray | Dict[str, np.ndarray], info: Dict) -> None:
            for i in range(self.batch_size):
                _observation = {k: v[i] for k, v in observation.items()} if isinstance(observation, Mapping) else observation[i]
                _info = {k: v[i] for k, v in info.items()}
                ep = Episode()
                ep.initialise(_observation, _info)
                self.ongoing_episodes.append(ep)

        def add(
            self,
            observation: np.ndarray | Dict[str, np.ndarray],
            reward: np.ndarray,
            action: np.ndarray,
            terminated: np.ndarray | bool,
            truncated: np.ndarray | bool,
            info: Dict,
        ) -> None:
            mask = np.logical_or(np.ravel(terminated), np.ravel(truncated))
            idxs = np.arange(self.batch_size)
            if "final_observation" in info:
                idxs = np.arange(self.batch_size)[mask]
                _o = info["final_observation"]
                for i in idxs:
                    _observation = {k: v[i] for k, v in _o.items()} if isinstance(_o, Mapping) else _o[i]
                    _info = info["final_info"][i]
                    self.ongoing_episodes[i].add(
                        _observation,
                        reward[i],
                        action[i],
                        terminated[i],
                        truncated[i],
                        _info,
                    )
                    self.completed_episodes.append(self.ongoing_episodes[i].get())
                    self.returns.append(np.sum(self.completed_episodes[-1]["reward"]))
                    self.lengths.append(len(self.completed_episodes[-1]["reward"]))
                    _observation = {k: v[i] for k, v in observation.items()} if isinstance(observation, Mapping) else observation[i]
                    _info = {k: v[i] for k, v in info.items()}
                    self.ongoing_episodes[i] = Episode()
                    self.ongoing_episodes[i].initialise(_observation, _info)
                # update remaining episodes
                idxs = np.arange(self.batch_size)[~mask]

            for i in idxs:
                _observation = {k: v[i] for k, v in observation.items()} if isinstance(observation, Mapping) else observation[i]
                _info = {k: v[i] for k, v in info.items() if not k.startswith("_")}
                self.ongoing_episodes[i].add(_observation, reward[i], action[i], terminated[i], truncated[i], _info)

        def __len__(self) -> int:
            return len(self.completed_episodes)

        def get(self) -> List[Dict[str, np.ndarray]]:
            return self.completed_episodes

elif Version(gymnasium.__version__) >= Version("1.0"):

    @dataclasses.dataclass
    class CollectEpisodes:
        batch_size: int
        completed_episodes: List[Episode] = dataclasses.field(default_factory=list)
        returns: List[float] = dataclasses.field(default_factory=list)
        lengths: List[int] = dataclasses.field(default_factory=list)
        ongoing_episodes: List[Episode] = dataclasses.field(default_factory=list)

        def initialise(self, observation: np.ndarray | Dict[str, np.ndarray], info: Dict) -> None:
            for i in range(self.batch_size):
                _observation = {k: v[i] for k, v in observation.items()} if isinstance(observation, Mapping) else observation[i]
                _info = {k: v[i] for k, v in info.items()}
                ep = Episode()
                ep.initialise(_observation, _info)
                self.ongoing_episodes.append(ep)
            self.has_autoreset = np.zeros(self.batch_size)

        def add(
            self,
            observation: np.ndarray | Dict[str, np.ndarray],
            reward: np.ndarray,
            action: np.ndarray,
            terminated: np.ndarray | bool,
            truncated: np.ndarray | bool,
            info: Dict,
        ) -> None:
            new_has_autoreset = np.logical_or(np.ravel(terminated), np.ravel(truncated))
            for i in range(self.batch_size):
                _info = {k: v[i] for k, v in info.items() if not k.startswith("_")}
                _observation = {k: v[i] for k, v in observation.items()} if isinstance(observation, Mapping) else observation[i]
                if self.has_autoreset[i]:
                    # last step was final (terminated or truncated), this step is simply for resetting the environment
                    # we don't need to store additional information
                    self.ongoing_episodes[i] = Episode()
                    self.ongoing_episodes[i].initialise(_observation, _info)
                else:
                    self.ongoing_episodes[i].add(
                        _observation,
                        reward[i],
                        action[i],
                        terminated[i],
                        truncated[i],
                        _info,
                    )
                    if new_has_autoreset[i]:
                        self.completed_episodes.append(self.ongoing_episodes[i].get())
                        self.returns.append(np.sum(self.completed_episodes[-1]["reward"]))
                        self.lengths.append(len(self.completed_episodes[-1]["reward"]))
                        self.ongoing_episodes[i] = Episode()  # this is not needed, it is just to be sure to not overwrite things
            self.has_autoreset = new_has_autoreset

        def __len__(self) -> int:
            return len(self.completed_episodes)

        def get(self) -> List[Dict[str, np.ndarray]]:
            return self.completed_episodes


else:
    raise ValueError("unsupported gymnasium version")
