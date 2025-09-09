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

    def __init__(self, width=800, height=600, num_players=1, num_obstacles=3, max_steps=10000):
        super(SoccerEnv, self).__init__()

        # 训练参数
        self.max_steps = max_steps  # 每个episode的最大步数
        self.current_step = 0       # 当前步数计数器
        self.steps_without_ball = 0  # 未触球步数计数器
        self.max_steps_without_ball = 5  # 最大允许未触球步数

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
        # 长时间未触球惩罚
        self.no_ball_penalty = -0.1  # 每步未触球的惩罚



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

    def reset(self):
        """重置环境"""
        # 重置计数器
        self.current_step = 0
        self.steps_without_ball = 0
        
        # 创建足球
        self.ball = Ball(self.width // 2, self.height // 2)

        # TODO: 修改球员的创建位置
        # 创建球员
        self.players = [Player(100 + i * 100, self.height // 2) for i in range(self.num_players)]
        
        # 创建障碍物
        self.obstacles = [Obstacle(500 + i * 50, self.height // 2) for i in range(self.num_obstacles)]
        
        # 重置奖励和完成标志
        self.reward = 0
        self.done = False
        
        # 重置上一帧状态（用于计算平滑奖励）
        for player in self.players:
            player.prev_vx = 0
            player.prev_vy = 0
            player.prev_angle = player.angle
        
        return self._get_observation()

    def step(self, action):
        """执行动作"""
        # 增加步数
        self.current_step += 1
        
        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            self.done = True
            return self._get_observation(), 0, True, {}
            
        # 重置奖励
        self.reward = 0
        
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

        # 处理碰撞
        self._handle_collisions()

        # 边界检查
        boundary_reward, self.done, boundary_info = self._check_boundaries()
        self.reward += boundary_reward

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
        self.reward += self._reward_2the_ball()
        self.reward += self._reward_smooth_movement()

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
        """
        检查并处理边界
        - 球出界：重置游戏并给予较大惩罚
        - 球员出界：允许一定出界范围，超出范围则扣分并重置位置
        """
        reward = 0
        game_reset = False
        
        # 球的边界检查 - 出界则重置游戏
        if (self.ball.x - self.ball.radius < 0 or 
            self.ball.x + self.ball.radius > self.width or
            self.ball.y - self.ball.radius < 0 or 
            self.ball.y + self.ball.radius > self.height):
            
            # 球出界，给予较大惩罚
            reward = -5.0
            game_reset = True
            return reward, game_reset, {}

        # 球员的边界检查 - 允许一定出界范围
        boundary_margin = 30  # 允许出界的像素范围
        for player in self.players:
            half_w = player.width / 2
            half_h = player.height / 2
            
            # 检查x轴边界
            if player.x - half_w < -boundary_margin:
                reward += -1.0  # 超出允许范围，扣分
                player.x = half_w
                player.vx = 0
            elif player.x + half_w > self.width + boundary_margin:
                reward += -1.0  # 超出允许范围，扣分
                player.x = self.width - half_w
                player.vx = 0
            
            # 检查y轴边界
            if player.y - half_h < -boundary_margin:
                reward += -1.0  # 超出允许范围，扣分
                player.y = half_h
                player.vy = 0
            elif player.y + half_h > self.height + boundary_margin:
                reward += -1.0  # 超出允许范围，扣分
                player.y = self.height - half_h
                player.vy = 0
                
            # 如果球员在边界内但接近边界，给予小惩罚
            elif (player.x - half_w < 0 or 
                  player.x + half_w > self.width or
                  player.y - half_h < 0 or 
                  player.y + half_h > self.height):
                reward += -0.1  # 小惩罚，因为球员在边界上但未超出允许范围
                
                # 将球员移回场内
                player.x = max(half_w, min(self.width - half_w, player.x))
                player.y = max(half_h, min(self.height - half_h, player.y))
                
        return reward, game_reset, {}

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

# ------------------------------奖励函数------------------------------ #

    def _reward_smooth_movement(self):
        """
        鼓励球员运动平滑的奖励函数
        奖励策略：
        1. 惩罚突然的转向（角度变化大）
        2. 惩罚突然的加速/减速
        3. 鼓励保持适中的速度
        
        :return: 奖励值
        """
        if not hasattr(self, 'players'):
            return 0.0
            
        total_reward = 0.0
        
        for player in self.players:
            # 如果这是第一步，初始化上一帧的状态
            if not hasattr(player, 'prev_vx'):
                player.prev_vx = player.vx
                player.prev_vy = player.vy
                player.prev_angle = player.angle
                return 0.0
                
            # 计算速度变化
            dvx = player.vx - player.prev_vx
            dvy = player.vy - player.prev_vy
            accel = (dvx**2 + dvy**2) ** 0.5
            
            # 计算角度变化（弧度）
            angle_change = abs(player.angle - player.prev_angle)
            angle_change = min(angle_change, 360 - angle_change)  # 取小角度
            
            # 计算当前速度
            speed = (player.vx**2 + player.vy**2) ** 0.5
            
            # 1. 惩罚突然的转向（角度变化大）
            angle_penalty = -0.1 * (angle_change / 10.0)  # 角度变化每10度惩罚0.1
            
            # 2. 惩罚突然的加速/减速
            accel_penalty = -0.0005 * (accel / 10.0)  # 加速度每10单位惩罚0.05
            
            # 3. 鼓励保持适中的速度（避免急停和全速）
            target_speed = 50.0  # 目标速度
            speed_diff = abs(speed - target_speed)
            speed_reward = -0.001 * (speed_diff / 10.0)  # 与目标速度的差距每10单位惩罚0.01
            
            # 更新上一帧状态
            player.prev_vx = player.vx
            player.prev_vy = player.vy
            player.prev_angle = player.angle
            
            # 总平滑奖励
            # smooth_reward = angle_penalty + accel_penalty + speed_reward
            smooth_reward = angle_penalty + speed_reward
            total_reward += smooth_reward
            
        return total_reward
        
    def _reward_2the_ball(self):
        """
        计算球员接近球的奖励
        奖励策略：
        1. 球员在球的左侧时，距离球越近奖励越高
        2. 球员在球的右侧时会受到惩罚
        3. 球员面向球移动时有额外奖励
        
        :return: 奖励值
        """
        if not hasattr(self, 'players') or not hasattr(self, 'ball'):
            return 0.0
            
        total_reward = 0.0
        
        for player in self.players:
            # 计算球员到球的向量
            dx = self.ball.x - player.x
            dy = self.ball.y - player.y
            distance_to_ball = (dx**2 + dy**2) ** 0.5
            
            # 归一化距离奖励 (0-1之间，越近值越大)
            max_distance = (self.width**2 + self.height**2) ** 0.5
            distance_reward = 1.0 - min(distance_to_ball / max_distance, 1.0)
            
            # 计算球员是否在球的左侧 (dx > 0 表示球员在球的左侧)
            if dx > 0:  # 球员在球的左侧
                position_reward = 1.0  # 在左侧有基础奖励
                # 距离越近奖励越高
                position_reward += (1.0 - (distance_to_ball / max_distance)) * 2.0
            else:  # 球员在球的右侧
                position_reward = -1.0  # 在右侧有惩罚
                # 距离越远惩罚越大
                position_reward -= 0.1 * (distance_to_ball / max_distance)
            
            # 计算球员速度方向与球方向的夹角
            if player.vx != 0 or player.vy != 0:
                # 球员速度方向
                player_direction = np.arctan2(player.vy, player.vx)
                # 球员到球的方向
                to_ball_direction = np.arctan2(dy, dx)
                # 计算方向差 (0-pi之间)
                angle_diff = abs(player_direction - to_ball_direction) % (2 * np.pi)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                # 方向奖励：面向球移动有额外奖励
                direction_reward = np.cos(angle_diff) * 0.5
            else:
                direction_reward = 0.0
            
            # 计算球员速度大小奖励 (鼓励快速移动)
            speed = (player.vx**2 + player.vy**2) ** 0.5
            speed_reward = min(speed / 100.0, 1.0) * 0.3  # 速度奖励上限为0.3
            
            # TODO: 总奖励 = 位置奖励 + 方向奖励 + 速度奖励
            player_reward = (position_reward * 0.5 +
                           direction_reward * 0.3 +
                           speed_reward) * 0.1  # 整体缩放因子
            total_reward += player_reward
        
        return total_reward
