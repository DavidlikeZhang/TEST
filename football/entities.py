import pygame
import numpy as np
from physics import collide_rect_circle, distance


class Entity:
    """基础实体类"""

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
        """更新实体位置"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.angular_velocity * dt
        # 保持角度在0-360之间
        self.angle = self.angle % 360

    def draw(self, screen):
        """绘制实体"""
        pass


class Ball(Entity):
    """足球类"""

    def __init__(self, x, y, radius=10):
        super().__init__(x, y, radius * 2, radius * 2, (255, 0, 0))  # 红色球
        self.radius = radius
        # 球的物理属性
        # TODO: 调节下面两个值
        self.mass = 1.0
        self.friction = 0.98  # 摩擦系数

    def update(self, dt):
        """更新球的位置，考虑摩擦力"""
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
        """绘制足球"""
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


class Player(Entity):
    """球员类"""

    def __init__(self, x, y, width=20, height=20):
        super().__init__(x, y, width, height, (0, 0, 255))  # 蓝色方块
        # 球员的物理属性
        # TODO: 调节下面几个值
        self.mass = 5.0
        self.max_speed = 100.0
        self.perception_range = 150.0  # 球员感知范围

        # 是否看到球
        self.can_see_ball = False

        self.max_kick = 500  # 最大踢球力度

    def can_perceive_ball(self, ball):
        """检查球员是否能感知到球"""
        # 计算球员朝向的向量
        direction_x = np.cos(np.radians(self.angle))
        direction_y = np.sin(np.radians(self.angle))

        # 计算球员到球的向量
        to_ball_x = ball.x - self.x
        to_ball_y = ball.y - self.y

        # 计算距离
        dist = distance((self.x, self.y), (ball.x, ball.y))

        # 计算点积，判断球是否在球员前方
        dot_product = direction_x * to_ball_x + direction_y * to_ball_y

        # 球在前方且在感知范围内
        if dot_product > 0 and dist < self.perception_range:
            self.can_see_ball = True
            return True

        self.can_see_ball = False
        return False

    def draw(self, screen):
        """绘制球员"""
        # 绘制方形球员
        rect = pygame.Rect(int(self.x - self.width / 2), int(self.y - self.height / 2),
                           self.width, self.height)
        pygame.draw.rect(screen, self.color, rect)

        # 绘制朝向指示线
        # TODO: 去掉这段代码，将绘制矩形改为基于四个旋转后的顶点绘制线段
        direction_x = np.cos(np.radians(self.angle)) * 20
        direction_y = np.sin(np.radians(self.angle)) * 20
        pygame.draw.line(screen, (255, 255, 255),
                         (int(self.x), int(self.y)),
                         (int(self.x + direction_x), int(self.y + direction_y)), 2)


        # 如果能看到球，绘制感知范围
        if self.can_see_ball:
            pygame.draw.circle(screen, (0, 255, 0, 50),
                               (int(self.x), int(self.y)),
                               int(self.perception_range), 1)


class Obstacle(Entity):
    """障碍物类"""

    def __init__(self, x, y, width=30, height=30):
        super().__init__(x, y, width, height, (150, 75, 0))  # 棕色障碍物
        self.target_x = x
        self.target_y = y
        self.speed = 50.0  # 障碍物移动速度

    def move_towards_ball(self, ball, goal):
        """移动障碍物挡在球和球门之间"""
        # 计算球和球门之间的向量
        to_goal_x = goal[0] - ball.x
        to_goal_y = goal[1] - ball.y

        # 标准化向量
        length = np.sqrt(to_goal_x ** 2 + to_goal_y ** 2)
        if length > 0:
            to_goal_x /= length
            to_goal_y /= length

        # 设定目标位置（在球和球门之间)
        self.target_x = ball.x + to_goal_x * 50
        self.target_y = ball.y + to_goal_y * 50

    def update(self, dt):
        """更新障碍物位置"""
        # 计算向目标移动的向量
        dx = self.target_x - self.x
        dy = self.target_y - self.y

        # 标准化向量
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 5:  # 如果距离目标较远，则移动
            dx = dx / length * self.speed * dt
            dy = dy / length * self.speed * dt

            self.x += dx
            self.y += dy

    def draw(self, screen):
        """绘制障碍物"""
        rect = pygame.Rect(int(self.x - self.width / 2), int(self.y - self.height / 2),
                           self.width, self.height)
        pygame.draw.rect(screen, self.color, rect)