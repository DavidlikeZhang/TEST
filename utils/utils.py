#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : Zhang Jinbin
# @time    : 2025/9/7 9:10
# @function: the script is used to do something.
# @version : V1


import pygame
import numpy as np


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