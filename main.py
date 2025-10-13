import os
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

# --- 1) Make base env (same as training) ---
def make_env():
    env = gym.make(
        "CarRacing-v3",
        render_mode="human",  # Enable human rendering for watching
        continuous=False
    )
    env = Monitor(env)
    env = GrayscaleObservation(env, keep_dim=True)  # (96, 96, 1) HWC
    return env

# It's good practice to wrap the environment in a VecEnv
vec_env = DummyVecEnv([make_env])

# Convert HWC -> CHW and stack 4 frames at the VecEnv level (same as training)
vec_env = VecTransposeImage(vec_env)       # (1, 96, 96)
vec_env = VecFrameStack(vec_env, 4)        # (4, 96, 96)

# 2. Load the saved model
TIMESTEPS = 500_000  # Match your training timesteps
model_path = f"models/PPO/carracing_ppo_{TIMESTEPS}.zip"
model = PPO.load(model_path, env=vec_env)

# model_path = f"models/DQN/carracing_dqn_{TIMESTEPS}.zip"
# model = DQN.load(model_path, env=vec_env)

# 3. Run the agent and watch it
episodes = 10
for ep in range(episodes):
    obs = vec_env.reset()  # Only one return value for VecEnv
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        # The env.render() is handled automatically by render_mode="human"

vec_env.close()