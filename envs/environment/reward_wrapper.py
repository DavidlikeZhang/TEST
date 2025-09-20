import gymnasium as gym
import numpy as np

class DefaultRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_num = getattr(env, "action_num", None)
        self.num_players = getattr(env, "num_players", None)
        self.reward = 0.0
        self.done = False

        # ----------------- 奖励参数 -----------------
        self.goal_reward = 10
        self.right_reward = 0.001
        self.see_ball_reward = 0.1
        self.time_penalty = -0.05
        self.bump_penalty = -0.2
        self.own_player_penalty = -20.0
        self.no_ball_penalty = -0.1

    def step(self, action):
        obs, terminated, truncated, info = None, False, False, {}

        # 先执行原始 env 的 step
        try:
            obs, _, terminated, truncated, info = self.env.step(action)
        except Exception:
            obs, terminated, truncated, info = self.env.step(action)  # 兼容一些 env 旧接口

        # 计算奖励
        reward = 0.0
        reward += self._calculate_boundary_reward()
        reward += self._calculate_goal_reward()
        reward += self._calculate_ball_movement_reward()
        reward += self._calculate_player_vision_reward()
        reward += self._reward_2the_ball()
        reward += self._reward_smooth_movement()
        reward += self.time_penalty  # 固定时间惩罚

        return obs, reward, terminated, truncated, info

    # ------------------------------奖励函数------------------------------ #
    def _calculate_boundary_reward(self):
        # 保证 env._check_boundaries 总是返回三个值
        if hasattr(self.env, "_check_boundaries"):
            res = self.env._check_boundaries()
            if isinstance(res, tuple) and len(res) == 3:
                boundary_reward, game_reset, _ = res
            else:
                print("Warning: Unexpected return value from _check_boundaries:", res)
                boundary_reward = 0.0
        else:
            boundary_reward = 0.0

        self.reward += boundary_reward
        return boundary_reward

    def _calculate_goal_reward(self):
        goal_reward = 0.0
        if hasattr(self.env, "ball") and hasattr(self.env, "width") and hasattr(self.env, "height"):
            try:
                from envs.physics.utils import is_goal
                goal = is_goal(self.env.ball, self.env.width - 50 + self.env.ball.radius, self.env.height)
                if goal == -1:
                    goal_reward += self.goal_reward
                elif goal == 1:
                    goal_reward += 10
            except Exception:
                goal_reward = 0.0
        self.reward += goal_reward
        return goal_reward

    def _calculate_ball_movement_reward(self):
        if not hasattr(self.env, "ball") or not hasattr(self.env, "width"):
            return 0.0
        dx = self.env.ball.x - self.env.width // 2
        ball_movement_reward = dx * self.right_reward
        self.reward += ball_movement_reward
        return ball_movement_reward

    def _calculate_player_vision_reward(self):
        if not hasattr(self.env, "players"):
            return 0.0
        vision_reward = 0.0
        for player in self.env.players:
            if getattr(player, "can_see_ball", False):
                vision_reward += self.see_ball_reward
        self.reward += vision_reward
        return vision_reward

    def _reward_smooth_movement(self):
        if not hasattr(self.env, "players"):
            return 0.0

        total_reward = 0.0
        for player in self.env.players:
            if not hasattr(player, 'prev_vx'):
                player.prev_vx = getattr(player, 'vx', 0.0)
                player.prev_vy = getattr(player, 'vy', 0.0)
                player.prev_angle = getattr(player, 'angle', 0.0)
                continue

            dvx = getattr(player, 'vx', 0.0) - player.prev_vx
            dvy = getattr(player, 'vy', 0.0) - player.prev_vy
            accel = (dvx**2 + dvy**2)**0.5

            angle_change = abs(getattr(player, 'angle', 0.0) - player.prev_angle)
            angle_change = min(angle_change, 360 - angle_change)

            speed = (getattr(player, 'vx', 0.0)**2 + getattr(player, 'vy', 0.0)**2)**0.5
            angle_penalty = -0.1 * (angle_change / 10.0)
            accel_penalty = -0.0005 * (accel / 10.0)
            target_speed = 50.0
            speed_diff = abs(speed - target_speed)
            speed_reward = -0.001 * (speed_diff / 10.0)

            player.prev_vx = getattr(player, 'vx', 0.0)
            player.prev_vy = getattr(player, 'vy', 0.0)
            player.prev_angle = getattr(player, 'angle', 0.0)

            smooth_reward = angle_penalty + accel_penalty + speed_reward
            total_reward += smooth_reward

        self.reward += total_reward
        return total_reward

    def _reward_2the_ball(self):
        if not hasattr(self.env, "players") or not hasattr(self.env, "ball"):
            return 0.0

        total_reward = 0.0
        for player in self.env.players:
            dx = self.env.ball.x - getattr(player, 'x', 0.0)
            dy = self.env.ball.y - getattr(player, 'y', 0.0)
            distance_to_ball = (dx**2 + dy**2)**0.5
            max_distance = (getattr(self.env, 'width', 1)**2 + getattr(self.env, 'height', 1)**2)**0.5
            distance_reward = 1.0 - min(distance_to_ball / max_distance, 1.0)

            position_reward = 1.0 if dx > 0 else -1.0
            position_reward += (1.0 - (distance_to_ball / max_distance)) * (2.0 if dx > 0 else -0.1)

            if getattr(player, 'vx', 0.0) != 0 or getattr(player, 'vy', 0.0) != 0:
                player_direction = np.arctan2(getattr(player, 'vy', 0.0), getattr(player, 'vx', 0.0))
                to_ball_direction = np.arctan2(dy, dx)
                angle_diff = abs(player_direction - to_ball_direction) % (2 * np.pi)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                direction_reward = np.cos(angle_diff) * 0.5
            else:
                direction_reward = 0.0

            speed = (getattr(player, 'vx', 0.0)**2 + getattr(player, 'vy', 0.0)**2)**0.5
            speed_reward = min(speed / 100.0, 1.0) * 0.3

            player_reward = (position_reward * 0.5 +
                             direction_reward * 0.3 +
                             speed_reward) * 0.1
            total_reward += player_reward

        self.reward += total_reward
        return total_reward
