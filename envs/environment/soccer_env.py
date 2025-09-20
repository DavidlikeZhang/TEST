# 分类: 环境模块
# 描述: 定义强化学习环境（SoccerEnv），包括状态空间、动作空间

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.entities.ball import Ball
from envs.entities.player import Player
from envs.entities.obstacle import Obstacle
from envs.physics.collision import collide_rect_rect, collide_rect_circle
from envs.physics.response import handle_collision
from envs.physics.utils import draw_field, is_goal


class SoccerEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width=800, height=600, num_players=1, num_obstacles=0, max_steps=10000):
        super(SoccerEnv, self).__init__()

        # 训练参数
        self.max_steps = max_steps  # 每个episode的最大步数
        self.current_step = 0       # 当前步数计数器
        self.steps_without_ball = 0  # 未触球步数计数器
        self.max_steps_without_ball = 5  # 最大允许未触球步数

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
        # 每个球员有7个状态: [x, y, angle, vx, vy, angular_velocity, can_see_ball]
        # 球有4个状态: [x, y, vx, vy]
        # 每个障碍物有2个状态: [x, y]
        player_obs = 7 * num_players  # 更新为7个状态
        ball_obs = 4
        obstacle_obs = 2 * num_obstacles

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.steps_without_ball = 0

        # 创建足球
        self.ball = Ball(self.width // 2, self.height // 2)

        # 创建球员
        self.players = [Player(100 + i * 100, self.height // 2) for i in range(self.num_players)]

        # 创建障碍物
        self.obstacles = [Obstacle(500 + i * 50, self.height // 2) for i in range(self.num_obstacles)]

        # 状态初始化
        for player in self.players:
            player.prev_vx = 0
            player.prev_vy = 0
            player.prev_angle = player.angle

        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        info = {}

        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            truncated = True
            return self._get_observation(), 0, terminated, truncated, info

        # 解析动作
        for i, player in enumerate(self.players):
            start_idx = i * self.action_num
            end_idx = start_idx + self.action_num
            player_action = action[start_idx:end_idx]
            forward_speed = player_action[0] * player.max_speed
            lateral_speed = player_action[1] * player.max_speed
            angular_vel = player_action[2] * 180  # 最大180度/秒

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

        # 碰撞处理
        collisions = self._handle_collisions()
        if collisions:
            info.update(collisions)

        # 检查进球
        goal = is_goal(self.ball, self.width-50+self.ball.radius, self.height)
        if goal == -1:
            terminated = True
            info["goal"] = "left"
        elif goal == 1:
            terminated = True
            info["goal"] = "right"

        # 边界检查
        boundary_reward, game_reset, boundary_info = self._check_boundaries()
        info = {}
        info.update(boundary_info)  
        if game_reset:
            terminated = True

        return self._get_observation(), 0.0, terminated, truncated, info

    def _get_observation(self):
        obs = []
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
        obs.extend([
            self.ball.x / self.width,
            self.ball.y / self.height,
            self.ball.vx / 300.0,
            self.ball.vy / 300.0
        ])
        for obstacle in self.obstacles:
            obs.extend([
                obstacle.x / self.width,
                obstacle.y / self.height
            ])
        return np.array(obs, dtype=np.float32)

    def _handle_collisions(self):
        info = {}
        # 球员与障碍物
        for i, player1 in enumerate(self.players):
            for obstacle in self.obstacles:
                if collide_rect_rect(player1, obstacle):
                    r = random.random()
                    if r < 0.5:
                        player1.vx *= 0.5
                        player1.vy *= 0.5
                    else:
                        player1.vx *= -0.05
                        player1.vy *= -0.05
                    info["collision_obstacle"] = True
            # 球员与球员
            for j, player2 in enumerate(self.players[i + 1:], i + 1):
                if collide_rect_rect(player1, player2):
                    player1.vx = 0
                    player1.vy = 0
                    info["collision_teammate"] = True
        # 球与球员
        for player in self.players:
            if collide_rect_circle(player, self.ball):
                handle_collision(self.ball, player)
                info["ball_touch"] = True
        # 球与障碍物
        for obstacle in self.obstacles:
            if collide_rect_circle(obstacle, self.ball):
                handle_collision(self.ball, obstacle)
                info["ball_hit_obstacle"] = True
        return info

# soccer_env.py 中

    def _check_boundaries(self):
        """检查球和球员是否出界，返回奖励、是否需要重置、info字典"""
        reward = 0.0
        game_reset = False
        info = {}

        # 球出界
        if (self.ball.x - self.ball.radius < 0 or
            self.ball.x + self.ball.radius > self.width or
            self.ball.y - self.ball.radius < 0 or
            self.ball.y + self.ball.radius > self.height):
            reward = -5.0
            game_reset = True
            info["ball_out"] = True

        # 球员出界
        boundary_margin = 30
        for player in self.players:
            half_w = player.width / 2
            half_h = player.height / 2
            if (player.x - half_w < -boundary_margin or 
                player.x + half_w > self.width + boundary_margin or
                player.y - half_h < -boundary_margin or 
                player.y + half_h > self.height + boundary_margin):
                info["player_out"] = True

        return reward, game_reset, info

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