from stable_baselines3 import PPO
from envs.environment.soccer_env import SoccerEnv

# 创建环境
env = SoccerEnv()

model = PPO.load("soccer_ppo_model.zip")

# 测试训练好的模型
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()