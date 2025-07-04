#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : Zhang Jinbin
# @time    : 2025/7/4 8:46
# @function: the script is used to do something.
# @version : V1

from GridVisualizer import GridVisualizer, CellType, ActionType, create_example_grid

import numpy as np
import matplotlib.pyplot as plt
import os


class ValueIteration:
    def __init__(self, grid, gamma=0.9, reward_target=1, reward_forbidden=-1, reward_normal=0, epsilon=0.01):
        self.grid = grid
        self.gamma = gamma
        self.reward_target = reward_target
        self.reward_forbidden = reward_forbidden
        self.reward_normal = reward_normal
        self.epsilon = epsilon

        self.action_type_num = ActionType.__len__()

        self.immediate_rewards = np.zeros((self.grid.m, self.grid.n, self.action_type_num))
        self.state_values = np.zeros((self.grid.m, self.grid.n))
        self.state_values_prev = np.full((self.grid.m, self.grid.n), -np.inf)
        self.action_values = np.zeros((self.grid.m, self.grid.n, self.action_type_num))

        self.max_action_value = np.zeros((self.grid.m, self.grid.n))
        self.max_action_value_idex = np.zeros((self.grid.m, self.grid.n))

        self.policy = np.full((self.grid.m, self.grid.n), ActionType.UP)

        self._init_immediate_rewards()

    def _init_immediate_rewards(self):
        '''
        init the immediate rewards
        '''
        for i in range(self.grid.m):
            for j in range(self.grid.n):
                # move up
                try:
                    s_a1 = self.grid.cell_types[i - 1, j]
                    if s_a1 == CellType.FORBIDDEN or i - 1 == -1:
                        self.immediate_rewards[i, j, 0] = self.reward_forbidden
                    elif s_a1 == CellType.TARGET:
                        self.immediate_rewards[i, j, 0] = self.reward_target
                    else:
                        self.immediate_rewards[i, j, 0] = self.reward_normal
                except IndexError:
                    self.immediate_rewards[i, j, 0] = self.reward_forbidden

                # move right
                try:
                    s_a2 = self.grid.cell_types[i, j + 1]
                    if s_a2 == CellType.TARGET:
                        self.immediate_rewards[i, j, 1] = self.reward_target
                    elif s_a2 == CellType.FORBIDDEN:
                        self.immediate_rewards[i, j, 1] = self.reward_forbidden
                    else:
                        self.immediate_rewards[i, j, 1] = self.reward_normal
                except IndexError:  # out of range
                    self.immediate_rewards[i, j, 1] = self.reward_forbidden

                # move down
                try:
                    s_a3 = self.grid.cell_types[i + 1, j]
                    if s_a3 == CellType.TARGET:
                        self.immediate_rewards[i, j, 2] = self.reward_target
                    elif s_a3 == CellType.FORBIDDEN:
                        self.immediate_rewards[i, j, 2] = self.reward_forbidden
                    else:
                        self.immediate_rewards[i, j, 2] = self.reward_normal
                except IndexError:
                    self.immediate_rewards[i, j, 2] = self.reward_forbidden

                # move left
                try:
                    s_a4 = self.grid.cell_types[i, j - 1]
                    if s_a4 == CellType.FORBIDDEN or j - 1 == -1:
                        self.immediate_rewards[i, j, 3] = self.reward_forbidden
                    elif s_a4 == CellType.TARGET:
                        self.immediate_rewards[i, j, 3] = self.reward_target
                    else:
                        self.immediate_rewards[i, j, 3] = self.reward_normal
                except IndexError:
                    self.immediate_rewards[i, j, 3] = self.reward_forbidden

                # stay
                s_a5 = self.grid.cell_types[i, j]
                if s_a5 == CellType.TARGET:
                    self.immediate_rewards[i, j, 4] = self.reward_target
                elif s_a5 == CellType.FORBIDDEN:
                    self.immediate_rewards[i, j, 4] = self.reward_forbidden
                else:
                    self.immediate_rewards[i, j, 4] = self.reward_normal


    def run(self) -> np.ndarray:
        """
        run the value interation algorithm
        :return: None
        """
        while(not self._is_converged()):
            self._step()
        return self.policy


    def _step(self):
        self.state_values_prev = self.state_values.copy()
        for i in range(self.grid.m):
            for j in range(self.grid.n):
                self._update_policy(i, j)
                self._calc_state_value(i, j)


    def _calc_state_value(self, i, j):
        """
        :param i: row
        :param j: column
        :return: v(i,j) = max_a q(i,j,a)
        """
        self.state_values[i, j] = self.max_action_value[i, j]


    def _calc_action_value(self, i, j, a):
        """
        q(i,j,a) = r(i,j,a) + gamma * v(i,j)
        :param i:
        :param j:
        :param a:
        :return:
        """
        self.action_values[i, j, a] = self.immediate_rewards[i, j, a] + self.gamma * self.state_values[i, j]

    def _update_policy(self, i, j):
        for a in range(self.action_type_num):
            self._calc_action_value(i, j, a)
        self.max_action_value_idex[i, j] = np.argmax(self.action_values[i, j])
        if self.max_action_value_idex[i, j] == 0:
            self.policy[i, j] = ActionType.UP
        elif self.max_action_value_idex[i, j] == 1:
            self.policy[i, j] = ActionType.RIGHT
        elif self.max_action_value_idex[i, j] == 2:
            self.policy[i, j] = ActionType.DOWN
        elif self.max_action_value_idex[i, j] == 3:
            self.policy[i, j] = ActionType.LEFT
        else:
            self.policy[i, j] = ActionType.STAY

    def _is_converged(self) -> bool:
        """
        :return: check if || v_k - v_k-1 || < epsilon
        """
        return np.all(np.abs(self.state_values - self.state_values_prev) <= self.epsilon)


if __name__ == "__main__":
    grid = create_example_grid()
    vi = ValueIteration(grid)
    grid.set_all_action(vi.run())

    fig, ax = grid.visualize()

    # Save the plot
    try:
        os.chdir("img")
    except OSError:
        os.mkdir("img")
        os.chdir("img")
    plt.savefig('via_grid_visualization.png', dpi=150, bbox_inches='tight')
    print("Grid visualization saved to /tmp/grid_visualization.png")

    # Also show if display is available
    try:
        plt.show()
    except:
        print("Display not available, plot saved to file instead")


