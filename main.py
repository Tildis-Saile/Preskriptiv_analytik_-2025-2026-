import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. Create the environment WITH rendering enabled
# IMPORTANT: Use render_mode="human"
env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
env = DummyVecEnv([lambda: env])

# 2. Load the saved model
TIMESTEPS = 500000
model_path = f"models/DQN/carracing_dqn_{TIMESTEPS}.zip"
model = DQN.load(model_path, env=env)

# 3. Run the agent and watch it
episodes = 10
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        # The env.render() is handled automatically by render_mode="human"

env.close()