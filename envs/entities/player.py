# 分类: 实体类模块
# 描述: 定义球员类（Player），继承自基础实体类（Entity）。

import pygame
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from envs.entities.base import Entity
from envs.entities.base import distance

class Player(Entity):
    """
    球员类
    """
    def __init__(self, x, y, width=20, height=20):
        super().__init__(x, y, width, height, (0, 0, 255))  # 蓝色方块
        self.mass = 5.0
        self.max_speed = 100.0
        self.perception_range = 150.0  # 球员感知范围
        self.can_see_ball = False
        self.max_kick = 500  # 最大踢球力度

    def can_perceive_ball(self, ball):
        """
        检查球员是否能感知到球
        """
        direction_x = np.cos(np.radians(self.angle))
        direction_y = np.sin(np.radians(self.angle))
        to_ball_x = ball.x - self.x
        to_ball_y = ball.y - self.y
        dist = distance((self.x, self.y), (ball.x, ball.y))
        dot_product = direction_x * to_ball_x + direction_y * to_ball_y
        if dot_product > 0 and dist < self.perception_range:
            self.can_see_ball = True
            return True
        self.can_see_ball = False
        return False

    def draw(self, screen):
        """
        绘制球员
        """
        rect = pygame.Rect(int(self.x - self.width / 2), int(self.y - self.height / 2),
                           self.width, self.height)
        pygame.draw.rect(screen, self.color, rect)
        direction_x = np.cos(np.radians(self.angle)) * 20
        direction_y = np.sin(np.radians(self.angle)) * 20
        pygame.draw.line(screen, (255, 255, 255),
                         (int(self.x), int(self.y)),
                         (int(self.x + direction_x), int(self.y + direction_y)), 2)
        if self.can_see_ball:
            pygame.draw.circle(screen, (0, 255, 0, 50),
                               (int(self.x), int(self.y)),
                               int(self.perception_range), 1)
