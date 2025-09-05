import numpy as np


def distance(p1, p2):
    """计算两点之间的距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def collide_rect_rect(rect1, rect2):
    """
    检测两个矩形是否碰撞
    rect1有顶点在rect2内，或者rect2有顶点在rect1内，发生碰撞，返回True
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
    rect: 有angle属性的实体
    circle: 有x, y, radius属性
    """
    # 将圆心转换到矩形的本地坐标系（逆向旋转-angle）
    angle = -rect.angle
    cx, cy = rect.x, rect.y
    dx, dy = circle.x - cx, circle.y - cy
    angle_rad = np.radians(angle)
    local_x = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    local_y = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    hw, hh = rect.width / 2, rect.height / 2
    # 1. 圆心在矩形范围内
    if -hw <= local_x <= hw and -hh <= local_y <= hh:
        return True
    # 2. 最近点在边上
    nearest_x = np.clip(local_x, -hw, hw)
    nearest_y = np.clip(local_y, -hh, hh)
    dist_sq = (local_x - nearest_x) ** 2 + (local_y - nearest_y) ** 2
    return dist_sq <= circle.radius ** 2


def handle_collision(entity1, entity2):
    """处理两个实体之间的碰撞（如球与旋转方形碰撞响应）"""
    # 假设entity1是球
    # 计算最近碰撞点
    angle = -entity2.angle
    cx, cy = entity2.x, entity2.y
    dx, dy = entity1.x - cx, entity1.y - cy
    angle_rad = np.radians(angle)
    local_x = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    local_y = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    hw, hh = entity2.width/2, entity2.height/2
    nearest_x = np.clip(local_x, -hw, hw)
    nearest_y = np.clip(local_y, -hh, hh)
    # 得到世界坐标下的碰撞点
    s = np.sin(-angle_rad)
    c = np.cos(-angle_rad)
    world_x = nearest_x * c - nearest_y * s + cx
    world_y = nearest_x * s + nearest_y * c + cy

    # 球的反弹方向
    nx = entity1.x - world_x
    ny = entity1.y - world_y
    norm = np.hypot(nx, ny)
    if norm == 0:
        nx, ny = 1, 0
    else:
        nx, ny = nx / norm, ny / norm

    # 简化反弹
    v_dot_n = entity1.vx * nx + entity1.vy * ny
    if v_dot_n < 0:
        entity1.vx -= 2 * v_dot_n * nx
        entity1.vy -= 2 * v_dot_n * ny
        # 摩擦和反弹衰减
        # TODO: 0.8这个值可能要调整
        entity1.vx *= 0.8
        entity1.vy *= 0.8


def rotate_point(px, py, cx, cy, angle_deg):
    """将dian(px, py)绕点(cx, cy)逆时针旋转angle_deg度"""
    angle_rad = np.radians(angle_deg)
    # angle_rad = np.deg0rad(angle_deg)
    s, c = np.sin(angle_rad), np.cos(angle_rad)
    dx = (px - cx) * c - (py - cy) * s
    dy = (px - cx) * s + (py - cy) * c
    return dx + cx, dy + cy


def get_rect_corners(entity):
    """
    获取任意实体（Player或Obstacle）旋转后的四个顶点坐标。
    返回顺序为：[左上, 右上, 右下, 左下]
    """
    hw, hh = entity.width / 2, entity.height / 2
    cx, cy = entity.x, entity.y  # 方形质心坐标
    angle = entity.angle # 角度制
    corners = [
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh)
    ]
    return [
        rotate_point(cx+dx, cy+dy, cx, cy, angle) for dx, dy in corners  # 返回一个四维数组
    ]


def point_in_rotated_rect(px, py, rect_corners):
    """
    判断点(px, py)是否在矩形rect_corners内
    rect_corners: 矩形旋转后顶点列表，顺序为[左上, 右上, 右下, 左下]
    """
    def check_point_side_edge(a, b, p):
        """
        使用叉乘（外积）判断点是否落在矩形内，a,b,p分别是二维坐标系下的三个点，计算b-a, p-a这两个向量的叉乘
        由于从a到b，绕矩形质心顺时针转动，作图可知：若p在矩形外内，则四次叉乘都必须是正数
        以上解释请结合以下的for循环来阅读
        """
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    inside = True
    for i in range(4):
        if check_point_side_edge(rect_corners[i], rect_corners[(i+1)%4], (px, py)) < 0:
            inside = False
            break
    return inside
