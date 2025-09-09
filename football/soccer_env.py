#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : Zhang Jinbin
# @time    : 2025/9/7 9:24
# @function: the script is used to do something.
# @version : V1
import random

# 注意，己方球员在仿真器中位于左方

import pygame
import numpy as np
import gym
from gym import spaces
from entities import Ball, Player, Obstacle
from physics import collide_rect_rect, collide_rect_circle, handle_collision
from utils import draw_field, is_goal


class SoccerEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width=800, height=600, num_players=1, num_obstacles=3):
        super(SoccerEnv, self).__init__()

        # TODO：修改或添加奖励
        # 进球奖励
        self.goal_reward = 10
        # 额外奖励：球向右球门移动
        self.right_reward = 0.001
        # 如果球员能看到球，给予奖励
        self.see_ball_reward = 0.1
        self.time_penalty = -0.01  # 时间惩罚
        # 摔倒惩罚
        self.bump_penalty = -0.2
        # 撞到队友了
        self.own_player_penalty = -20.0



        self.width = width
        self.height = height
        self.num_players = num_players
        self.num_obstacles = num_obstacles

        # 球门位置
        self.left_goal = (0, self.height // 2)
        self.right_goal = (self.width, self.height // 2)

        # 定义动作空间: [前后速度, 左右速度, 自转速度]
        # 每个球员有3个动作
        self.action_num = 3
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]*num_players),
            high=np.array([1, 1, 1]*num_players),
            dtype=np.float32
        )

        # 定义观测空间：
        # 每个球员有6个状态: [x, y, angle, vx, vy, angular_velocity]
        # 球有4个状态: [x, y, vx, vy]
        # 每个障碍物有2个状态: [x, y]
        player_obs = 6*num_players
        ball_obs = 4
        obstacle_obs = 2*num_obstacles

        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(player_obs + ball_obs + obstacle_obs,),
            dtype=np.float32
        )

        # 初始化Pygame
        self.screen = None
        self.clock = None
        self.isopen = True

        # 初始化实体
        self.reset()

    def reset(self):
        """重置环境"""
        # 创建足球
        self.ball = Ball(self.width // 2, self.height // 2)

        # 创建球员
        self.players = []
        for i in range(self.num_players):
            # TODO: 修改初始位置
            x = self.width // 4 + i * 50
            y = self.height // 2 + i * 30 - 30
            player = Player(x, y)
            self.players.append(player)

        # 创建障碍物
        self.obstacles = []
        for i in range(self.num_players):
            # TODO: 修改初始位置
            x = self.width * 3 // 4 - i * 40
            y = self.height // 2 + i * 40 - 40
            obstacle = Obstacle(x, y)
            self.obstacles.append(obstacle)

        # 游戏状态
        self.done = False  # 是否结束
        self.reward = 0
        self.steps = 0
        # self.max_steps = 1000  # TODO: 应该能走无限步

        return self._get_observation()

    def step(self, action):
        """执行动作"""
        self.reward = 0  # 每个step的奖励

        for i, player in enumerate(self.players):
            # 获取这个球员的动作
            act_idx = i*self.action_num
            forward_speed = action[act_idx] * player.max_speed
            lateral_speed = action[act_idx + 1] * player.max_speed
            angular_vel = action[act_idx + 2] * 180  # 最大180度/秒

            # 计算球员的速度向量
            angle_rad = np.radians(player.angle)
            player.vx = np.cos(angle_rad) * forward_speed - np.sin(angle_rad) * lateral_speed
            player.vy = np.sin(angle_rad) * forward_speed + np.cos(angle_rad) * lateral_speed
            player.angular_velocity = angular_vel

            # 更新球员是否能看到球
            player.can_perceive_ball(self.ball)

        # 更新障碍物
        for obstacle in self.obstacles:
            obstacle.move_towards_ball(self.ball, self.right_goal)

        dt = 0.1 # 时间步长，没什么好调的
        # 更新所有实体位置
        for player in self.players:
            player.update(dt)

        for obstacle in self.obstacles:
            obstacle.update(dt)

        self.ball.update(dt)

        # 处理碰撞
        self._handle_collisions()

        # 边界检查
        self._check_boundaries()

        # 检查是否进球
        goal = is_goal(self.ball, self.width, self.height)
        if goal == -1:  # 左队进球
            self.reward += -self.goal_reward
            self.done = True
        elif goal == 1:  # TODO：右队进球，我们默认是右队，实际部署是也要这样
            self.reward = 10
            self.done = True

        # 额外奖励：球向右球门移动
        dx = self.ball.x - self.width // 2
        self.reward += dx * self.right_reward

        # 如果球员能看到球，给予奖励
        for player in self.players:
            if player.can_see_ball:
                self.reward += self.see_ball_reward

        self.reward += self.time_penalty

        return self._get_observation(), self.reward, self.done, {}

    def _get_observation(self):
        """获取环境观察"""
        obs = []

        # 球员状态
        for player in self.players:
            obs.extend([
                player.x / self.width,
                player.y / self.height,
                player.angle / 360.0,
                player.vx / player.max_speed,
                player.vy / player.max_speed,
                player.angular_velocity / 180.0,
                float(player.can_see_ball)
            ])

        # 球状态
        obs.extend([
            self.ball.x / self.width,
            self.ball.y / self.height,
            self.ball.vx / 300.0,
            self.ball.vy / 300.0
        ])

        # 障碍物状态
        for obstacle in self.obstacles:
            obs.extend([
                obstacle.x / self.width,
                obstacle.y / self.height
            ])

        return np.array(obs, dtype=np.float32)

    def _handle_collisions(self):
        """处理所有碰撞"""
        # 球员与球员碰撞
        for i, player1 in enumerate(self.players):
            # 球员与障碍物碰撞
            for obstacle in self.obstacles:
                if collide_rect_rect(player1, obstacle):
                    # 简单的反弹
                    # TODO： 这里值得探讨，撞到敌方是否要惩罚，还是要鼓励？
                    # 百分之50的几率撞倒对方并减速，百分之50的几率被撞倒， TODO：0.5要修改
                    r = random.random()
                    if r < 0.5:
                        player1.vx = player1.vx * 0.5
                        player1.vy = player1.vy * 0.5
                    else:
                        player1.vx = -player1.vx * 0.05
                        player1.vy = -player1.vy * 0.05
                        self.reward += -self.bump_penalty

            # 球员与其他球员碰撞
            for j, player2 in enumerate(self.players[i + 1:], i + 1):
                if collide_rect_rect(player1, player2):
                    # 速度变成0并惩罚
                    player1.vx = 0
                    player1.vy = 0
                    self.reward += self.own_player_penalty

        # 球与球员碰撞
        for player in self.players:
            if collide_rect_circle(player, self.ball):
                handle_collision(self.ball, player)

        # 球与障碍物碰撞
        for obstacle in self.obstacles:
            if collide_rect_circle(obstacle, self.ball):
                handle_collision(self.ball, obstacle)

    def _check_boundaries(self):
        """检查并处理边界"""
        # 球的边界检查
        if self.ball.x - self.ball.radius < 0:
            self.ball.x = self.ball.radius
            self.ball.vx = -self.ball.vx * 0.8
        elif self.ball.x + self.ball.radius > self.width:
            self.ball.x = self.width - self.ball.radius
            self.ball.vx = -self.ball.vx * 0.8

        if self.ball.y - self.ball.radius < 0:
            self.ball.y = self.ball.radius
            self.ball.vy = -self.ball.vy * 0.8
        elif self.ball.y + self.ball.radius > self.height:
            self.ball.y = self.height - self.ball.radius
            self.ball.vy = -self.ball.vy * 0.8

        # 球员的边界检查
        for player in self.players:
            half_w = player.width / 2
            half_h = player.height / 2

            if player.x - half_w < 0:
                player.x = half_w
                player.vx = 0
            elif player.x + half_w > self.width:
                player.x = self.width - half_w
                player.vx = 0

            if player.y - half_h < 0:
                player.y = half_h
                player.vy = 0
            elif player.y + half_h > self.height:
                player.y = self.height - half_h
                player.vy = 0

        # 障碍物的边界检查
        for obstacle in self.obstacles:
            half_w = obstacle.width / 2
            half_h = obstacle.height / 2

            if obstacle.x - half_w < 0:
                obstacle.x = half_w
            elif obstacle.x + half_w > self.width:
                obstacle.x = self.width - half_w

            if obstacle.y - half_h < 0:
                obstacle.y = half_h
            elif obstacle.y + half_h > self.height:
                obstacle.y = self.height - half_h

    def render(self, mode='human'):
        """渲染环境"""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Soccer RL Environment')

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # 绘制足球场
        draw_field(self.screen, self.width, self.height)

        # 绘制球
        self.ball.draw(self.screen)

        # 绘制障碍物
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        # 绘制球员
        for player in self.players:
            player.draw(self.screen)

        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(30)
            return None
        elif mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False