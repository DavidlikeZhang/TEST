#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : Zhang Jinbin
# @time    : 2025/9/7 11:49
# @function: the script is used to do something.
# @version : V1

# 注意！！！！！！！！！，使用英文输入法，否则控制不了



import pygame
import numpy as np
from soccer_env import SoccerEnv


def main():
    # 创建环境
    env = SoccerEnv(width=800, height=600, num_players=3)
    action_n = env.action_num
    obs = env.reset()

    # 手动控制参数
    key_actions = {
        pygame.K_w: [1, 0, 0],  # 前进
        pygame.K_s: [-1, 0, 0],  # 后退
        pygame.K_a: [0, -1, 0],  # 左移
        pygame.K_d: [0, 1, 0],  # 右移
        pygame.K_q: [0, 0, -1],  # 左转
        pygame.K_e: [0, 0, 1],  # 右转
    }

    # 主循环
    running = True
    while running:
        # 渲染环境
        env.render()

        # 输入动作初始化
        action = np.zeros(env.action_space.shape)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()  # 返回一个布尔序列
        # if keys[pygame.K_SPACE]:
        #     print("空格键被按下")
        for key, act in key_actions.items():
            if keys[key]:
                action[:action_n] += act

        # 对第一个球员应用动作，其他球员随机行动
        for i in range(1, env.num_players):
            idx = i * action_n # 目前，每个球员有三个动作
            action[idx:idx + action_n] = env.action_space.sample()[idx:idx + action_n]

        # 执行动作
        obs, reward, done, _ = env.step(action)

        # 显示一些信息
        # print(f"Reward: {reward:.2f}")

        if done:
            print("Episode finished!")
            obs = env.reset()

        # 控制帧率
        pygame.time.delay(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()