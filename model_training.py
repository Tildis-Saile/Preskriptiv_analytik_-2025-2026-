import os
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

# --- 1) Make base env (discrete actions for DQN/PPO) ---
def make_env():
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",   # faster for training; use "human" to watch
        continuous=False
    )
    env = Monitor(env)
    env = GrayscaleObservation(env, keep_dim=True)  # (96, 96, 1) HWC
    return env

# It's good practice to wrap the environment in a VecEnv
vec_env = DummyVecEnv([make_env])

# Convert HWC -> CHW and stack 4 frames at the VecEnv level
vec_env = VecTransposeImage(vec_env)       # (1, 96, 96)
vec_env = VecFrameStack(vec_env, 4)        # (4, 96, 96)

# Quick sanity check (outside VecEnv)
_tmp = make_env()
print("Single-env observation space (no stack):", _tmp.observation_space)
# Expect: Box(0, 255, (96, 96, 1), uint8)
_tmp.close()

os.makedirs("models/DQN", exist_ok=True)
os.makedirs("models/PPO", exist_ok=True)

TIMESTEPS = 500_000
# TIMESTEPS = 5_000

# --- Method 1: DQN ---
model_dqn = DQN(
    "CnnPolicy",
    vec_env,
    verbose=1,
    buffer_size=50_000,
    learning_rate=1e-4,
    batch_size=32,
    learning_starts=1_000,
    target_update_interval=1_000,
    tensorboard_log="./carracing_tensorboard/"
)


# model_dqn.learn(total_timesteps=TIMESTEPS)  # fresh run → omit reset_num_timesteps
# model_dqn.save(f"models/DQN/carracing_dqn_{TIMESTEPS}.zip")
print("DQN training complete and model saved.")

# --- Method 2: PPO ---
# You can reuse the same vec_env; don't close it until you're totally done.
model_ppo = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    tensorboard_log="./carracing_tensorboard/"
)
model_ppo.learn(total_timesteps=TIMESTEPS)
model_ppo.save(f"models/PPO/carracing_ppo_{TIMESTEPS}.zip")
print("PPO training complete and model saved.")

# Close AFTER all training is done
vec_env.close()
