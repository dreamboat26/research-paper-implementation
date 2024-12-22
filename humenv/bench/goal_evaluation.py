# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Sequence, Callable, List
import dataclasses
import numpy as np
from humenv import make_humenv
from humenv.bench.utils.metrics import get_episode_goal_stats
from humenv.bench import convert_dict_of_lists
from humenv.bench.gym_utils.rollouts import rollout
import tqdm
import gymnasium


@dataclasses.dataclass(kw_only=True)
class GoalEvaluation:
    goals: dict[np.ndarray]  # Dict of goal poses to reach
    motion_base_path: str | None = None
    motions: str | List[str] | None = None
    num_contexts: int = 1
    num_episodes: int = 1
    # environment parameters
    num_envs: int = 1
    vectorization_mode: str = "async"
    wrappers: Sequence[Callable[[gymnasium.Env], gymnasium.Wrapper]] = dataclasses.field(
        default_factory=lambda: [gymnasium.wrappers.FlattenObservation]
    )
    env_kwargs: dict = dataclasses.field(default_factory=dict)

    def run(self, agent: Any) -> Dict[str, Any]:
        metrics = {}
        self.env_kwargs["task"] = None
        penv, manager = make_humenv(
            num_envs=self.num_envs,
            vectorization_mode=self.vectorization_mode,
            motions=self.motions,
            motion_base_path=self.motion_base_path,
            wrappers=self.wrappers,
            **self.env_kwargs,
        )
        pbar = tqdm.tqdm(self.goals.items())
        for goal_name, goal_pose in pbar:
            pbar.set_description(f"Goal {goal_name}")
            local_stats = []
            for _ in range(self.num_contexts):
                ctx = agent.goal_inference(goal_pose)
                ctx = [None] * self.num_envs if ctx is None else ctx.repeat(self.num_envs, 1)
                st, episode = rollout(
                    penv,
                    agent=agent,
                    num_episodes=self.num_episodes,
                    ctx=ctx,
                )  # return statistics and episodes

                # Stats calculated over whole episode:
                for ep in episode:
                    ep["goal"] = goal_pose
                st_ep = get_episode_goal_stats(episode, device="cpu")
                st.update(st_ep)

                local_stats.append(st)
            local_stats = convert_dict_of_lists(local_stats)
            metrics[goal_name] = local_stats
        penv.close()
        if manager is not None:
            manager.shutdown()
        return metrics
