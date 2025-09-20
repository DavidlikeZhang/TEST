from stable_baselines3 import PPO
from envs.environment.soccer_env import SoccerEnv
from envs.environment.reward_wrapper import DefaultRewardWrapper

# 创建环境
env = DefaultRewardWrapper(SoccerEnv())

# 创建并训练模型
model = PPO("MlpPolicy", env, verbose=1, device='cuda',
            learning_rate=0.0003,  # 调整学习率
            gamma=0.99,  # 折扣因子
            n_steps=2048,  # 每次更新的步数
            ent_coef=0.01,  # 增加探索度的熵系数
            clip_range=0.3)  # 增大剪辑范围
model.learn(total_timesteps=100000)

# 保存模型
model.save("soccer_ppo_model")

# 测试训练好的模型
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()
