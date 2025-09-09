from stable_baselines3 import PPO
from soccer_env import SoccerEnv

# 创建环境
env = SoccerEnv()

# 创建并训练模型
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# 保存模型
model.save("soccer_ppo_model")

# 测试训练好的模型
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()