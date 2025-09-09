import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pygame
from envs.entities.base import Entity

# 分类: 实体类模块
# 描述: 定义足球类（Ball），继承自基础实体类（Entity）。

class Ball(Entity):
    """
    足球类
    """
    def __init__(self, x, y, radius=10):
        super().__init__(x, y, radius * 2, radius * 2, (255, 0, 0))  # 红色球
        self.radius = radius
        self.mass = 50.0
        self.friction = 0.88  # 摩擦系数

    def update(self, dt):
        """
        更新球的位置，考虑摩擦力
        """
        super().update(dt)
        # 应用摩擦力
        self.vx *= self.friction
        self.vy *= self.friction

        # 如果速度很小，停止移动
        if abs(self.vx) < 0.1:
            self.vx = 0
        if abs(self.vy) < 0.1:
            self.vy = 0

    def draw(self, screen):
        """
        绘制足球
        """
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
