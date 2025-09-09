import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pygame
import numpy as np

class Entity:
    """
    基础实体类
    """
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.vx = 0
        self.vy = 0
        self.angle = 0
        self.angular_velocity = 0

    def update(self, dt):
        """
        更新实体位置
        """
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.angular_velocity * dt
        # 保持角度在0-360之间
        self.angle = self.angle % 360

    def draw(self, screen):
        """
        绘制实体
        """
        pass

def distance(p1, p2):
    """计算两点之间的距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
