# 分类: 物理模块
# 描述: 包含碰撞检测相关函数，例如矩形与矩形、矩形与圆形的碰撞检测。
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from envs.physics.utils import rotate_point, get_rect_corners, point_in_rotated_rect

def collide_rect_rect(rect1, rect2):
    """
    检测两个矩形是否碰撞
    """
    corners1 = get_rect_corners(rect1)
    corners2 = get_rect_corners(rect2)
    for p1 in corners1:
        if point_in_rotated_rect(p1[0], p1[1], corners2):
            return True
    for p2 in corners2:
        if point_in_rotated_rect(p2[0], p2[1], corners1):
            return True
    return False

def collide_rect_circle(rect, circle):
    """
    检测旋转矩形与圆形是否碰撞
    """
    angle = -rect.angle
    cx, cy = rect.x, rect.y
    dx, dy = circle.x - cx, circle.y - cy
    angle_rad = np.radians(angle)
    local_x = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    local_y = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    hw, hh = rect.width / 2, rect.height / 2
    if -hw <= local_x <= hw and -hh <= local_y <= hh:
        return True
    nearest_x = np.clip(local_x, -hw, hw)
    nearest_y = np.clip(local_y, -hh, hh)
    dist_sq = (local_x - nearest_x) ** 2 + (local_y - nearest_y) ** 2
    return dist_sq <= circle.radius ** 2
