#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : Zhang Jinbin
# @time    : 2025/7/5 21:16
# @function: the script is used to do something.
# @version : V1

import numpy as np

from typing import Optional


class GPI:
    """
    General Policy Iteration
    """
    def __init__(self, env, truncate_iteration: Optional[int] = None, gamma=0.9, epsilon=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        self.state_num = env.get_state_num()
        self.action_num = env.get_action_num()

        self.state_values = np.zeros(self.state_num)
        self.state_values_prev = np.zeros(self.state_num)
        self.policy = np.zeros(self.state_num)

        self.iteration_num = 0
        pass

    def run(self):
        while not self._is_converged():
            self._step()
            self.iteration_num += 1

    def _step(self):
        self._policy_evaluation()
        self._policy_improvement()

    def _policy_evaluation(self):
        pass

    def _policy_improvement(self):
        pass

    def _is_converged(self) -> bool:
        """
        :return: check if || v_k - v_k-1 || < epsilon
        """
        return np.all(np.abs(self.state_values - self.state_values_prev) <= self.epsilon)