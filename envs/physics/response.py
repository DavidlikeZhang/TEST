# 分类: 物理模块
# 描述: 包含碰撞响应相关函数，例如处理实体之间的碰撞反应。

import numpy as np

def handle_collision(entity1, entity2):
    """
    处理两个实体之间的碰撞
    """
    angle = -entity2.angle
    cx, cy = entity2.x, entity2.y
    dx, dy = entity1.x - cx, entity1.y - cy
    angle_rad = np.radians(angle)
    local_x = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    local_y = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    hw, hh = entity2.width / 2, entity2.height / 2
    nearest_x = np.clip(local_x, -hw, hw)
    nearest_y = np.clip(local_y, -hh, hh)
    s = np.sin(-angle_rad)
    c = np.cos(-angle_rad)
    world_x = nearest_x * c - nearest_y * s + cx
    world_y = nearest_x * s + nearest_y * c + cy
    nx = entity1.x - world_x
    ny = entity1.y - world_y
    norm = np.hypot(nx, ny)
    if norm == 0:
        nx, ny = 1, 0
    else:
        nx, ny = nx / norm, ny / norm
    rvx = entity1.vx - entity2.vx
    rvy = entity1.vy - entity2.vy
    v_dot_n = rvx * nx + rvy * ny
    if v_dot_n < 0:
        restitution = 0.4
        impulse = -(1 + restitution) * v_dot_n
        entity1.vx += impulse * nx
        entity1.vy += impulse * ny
    overlap = entity1.radius - norm
    if overlap > 0:
        entity1.x += nx * overlap
        entity1.y += ny * overlap
