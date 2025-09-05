#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : Zhang Jinbin
# @time    : 2025/7/28 17:18
# @function: the script is used to do something.
# @version : V1

from test_compute_reward_config import Cfg
class A:
    def _reward_reward_1(self):
        print("这是奖励1hhhh")
        return 1

    def _reward_reward_2(self):
        print("这是奖励2")
        return 2


a = A()
cfg = Cfg()

rew_buf = 0
rew_termination = -100


def class_to_dict(obj) -> dict:
    if not  hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


reward_scales = class_to_dict(cfg.reward.scales)

# prepare reward functions
for key in list(reward_scales.keys()):
    scale = reward_scales[key]
    if scale == 0:
        reward_scales.pop(key)
    # prepare list of functions
    reward_functions = []
    reward_names = []
    for name, scale in reward_scales.items():
        if name == "termination":
            continue
        reward_names.append(name)
        name = "_reward_" + name
        reward_functions.append(getattr(a,name))

# compute reward
for i in range(len(reward_functions)):
    name = reward_names[i]
    rew_buf += reward_functions[i]() * reward_scales[name]
    if "termination" in reward_scales:
        rew_buf += rew_termination * reward_scales["termination"]

print(f"rew_buf: {rew_buf}")
