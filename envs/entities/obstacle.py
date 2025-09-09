import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pygame
import numpy as np
from envs.entities.base import Entity

# 分类: 实体类模块
# 描述: 定义障碍物类（Obstacle），继承自基础实体类（Entity）。

class Obstacle(Entity):
    """
    障碍物类
    """
    def __init__(self, x, y, width=30, height=30):
        super().__init__(x, y, width, height, (150, 75, 0))  # 棕色障碍物
        self.target_x = x
        self.target_y = y
        self.speed = 20.0

    def move_towards_ball(self, ball, goal):
        """
        移动障碍物挡在球和球门之间
        """
        to_goal_x = goal[0] - ball.x
        to_goal_y = goal[1] - ball.y
        length = np.sqrt(to_goal_x ** 2 + to_goal_y ** 2)
        if length > 0:
            to_goal_x /= length
            to_goal_y /= length
        self.target_x = ball.x + to_goal_x * 50
        self.target_y = ball.y + to_goal_y * 50

    def update(self, dt):
        """
        更新障碍物位置
        """
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 5:
            dx = dx / length * self.speed * dt
            dy = dy / length * self.speed * dt
            self.x += dx
            self.y += dy

    def draw(self, screen):
        """
        绘制障碍物
        """
        rect = pygame.Rect(int(self.x - self.width / 2), int(self.y - self.height / 2),
                           self.width, self.height)
        pygame.draw.rect(screen, self.color, rect)
