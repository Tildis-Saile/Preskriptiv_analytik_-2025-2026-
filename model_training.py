import gymnasium as gym
import cv2
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Create the base environment with discrete actions
env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

# 2. Apply Frame Stacking

print("Observation space shape:", env.observation_space.shape)
# Expected output: Observation space shape: (4, 96, 96)

# It's good practice to wrap the environment in a VecEnv
vec_env = DummyVecEnv([lambda: env])

# --- Method 1: DQN ---
# 'CnnPolicy' tells the model to use a CNN to process the image observations.
model_dqn = DQN(
    'CnnPolicy',
    vec_env,
    verbose=1,
    buffer_size=50000, # Size of the replay buffer
    learning_rate=1e-4,
    batch_size=32,
    learning_starts=1000, # Number of steps to collect before training starts
    target_update_interval=1000,
    tensorboard_log="./carracing_tensorboard/"
)

TIMESTEPS = 500000
model_dqn.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

# 4. Save the model
model_dqn.save(f"models/DQN/carracing_dqn_{TIMESTEPS}.zip")
print("Training complete and model saved.")

env.close()

# --- Method 2: PPO ---
model_ppo = PPO(
    'CnnPolicy',
    vec_env,
    verbose=1,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    tensorboard_log="./carracing_tensorboard/"
)
# model_ppo.learn(total_timesteps=500000) # Train for 500k steps