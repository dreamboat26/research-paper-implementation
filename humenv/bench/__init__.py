# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import collections


def convert_dict_of_lists(data):
    dict_of_lists = collections.defaultdict(list)
    for item in data:
        for key, value in item.items():
            if isinstance(value, collections.abc.Iterable):
                dict_of_lists[key].extend(value)
            else:
                dict_of_lists[key].append(value)
    return dict(dict_of_lists)


from humenv.bench.goal_evaluation import GoalEvaluation
from humenv.bench.reward_evaluation import RewardEvaluation
from humenv.bench.tracking_evaluation import TrackingEvaluation
