# 分类: 物理模块
# 描述: 提供物理计算的工具函数，例如点旋转、矩形顶点计算等。

import numpy as np
import pygame

def is_goal(ball, width, height):
    """检测是否进球"""
    goal_height = 200
    goal_y = (height - goal_height) // 2

    # 左侧球门
    if ball.x - ball.radius <= 0 and goal_y <= ball.y <= goal_y + goal_height:
        return 1  # 右队进球

    # 右侧球门
    if ball.x + ball.radius >= width and goal_y <= ball.y <= goal_y + goal_height:
        return -1  # 左队进球

    return 0  # 没有进球

def draw_field(screen, width, height):
    """
    绘制足球场
    """
    screen.fill((40, 180, 40))

    # 白色边线
    border_thickness = 4
    pygame.draw.rect(screen, (255, 255, 255),
                     (border_thickness // 2, border_thickness // 2,
                      width - border_thickness, height - border_thickness),
                     border_thickness)

    # 中线
    pygame.draw.line(screen, (255, 255, 255),
                     (width // 2, 0), (width // 2, height), border_thickness)

    # 中圈
    pygame.draw.circle(screen, (255, 255, 255),
                       (width // 2, height // 2), 60, border_thickness)
    pygame.draw.circle(screen, (255, 255, 255),
                       (width // 2, height // 2), 3, 0)

    # 球门区域（左侧）
    goal_width = 100
    goal_height = 200
    goal_y = (height - goal_height) // 2
    pygame.draw.rect(screen, (255, 255, 255),
                     (0, goal_y, 50, goal_height),
                     border_thickness)

    # 球门区域（右侧）
    pygame.draw.rect(screen, (255, 255, 255),
                     (width - 50, goal_y, 50, goal_height),
                     border_thickness)

    # 绘制横向白线（草坪纹理）
    stripe_spacing = 40
    for y in range(0, height, stripe_spacing):
        pygame.draw.line(screen, (220, 255, 220),
                         (0, y), (width, y), 1)

def rotate_point(px, py, cx, cy, angle_deg):
    """
    将点(px, py)绕点(cx, cy)逆时针旋转angle_deg度
    """
    angle_rad = np.radians(angle_deg)
    s, c = np.sin(angle_rad), np.cos(angle_rad)
    dx = (px - cx) * c - (py - cy) * s
    dy = (px - cx) * s + (py - cy) * c
    return dx + cx, dy + cy

def get_rect_corners(entity):
    """
    获取任意实体旋转后的四个顶点坐标
    """
    hw, hh = entity.width / 2, entity.height / 2
    cx, cy = entity.x, entity.y
    angle = entity.angle
    corners = [
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh)
    ]
    return [
        rotate_point(cx + dx, cy + dy, cx, cy, angle) for dx, dy in corners
    ]

def point_in_rotated_rect(px, py, rect_corners):
    """
    判断点(px, py)是否在矩形rect_corners内
    """
    def check_point_side_edge(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    inside = True
    for i in range(4):
        if check_point_side_edge(rect_corners[i], rect_corners[(i + 1) % 4], (px, py)) < 0:
            inside = False
            break
    return inside
