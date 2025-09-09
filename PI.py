#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : Zhang Jinbin
# @time    : 2025/7/5 21:36
# @function: the script is used to do something.
# @version : V1

import numpy as np
from typing import Optional

from GPI import GPI


class PI(GPI):
    def __init__(self, env, truncate_iteration: Optional[int] = None, gamma=0.9, epsilon=0.01):
        super(PI, self).__init__(env, truncate_iteration, gamma, epsilon)

    def _policy_evaluation(self):
        pass

    def _policy_improvement(self):
        pass

