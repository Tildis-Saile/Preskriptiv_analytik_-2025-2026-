import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

def make_env(render_mode="rgb_array"):
    """Create environment with same preprocessing as training"""
    env = gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        continuous=False
    )
    env = Monitor(env)
    env = GrayscaleObservation(env, keep_dim=True)
    return env

def evaluate_model(model, vec_env, num_episodes=50):
    """Evaluate a model and return episode rewards"""
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs = vec_env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:  # Safety limit
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    return episode_rewards, episode_lengths

def main():
    print("=== Model Comparison: DQN vs PPO (500,000 timesteps) ===")
    
    # Create environments
    print("Creating environments...")
    vec_env_dqn = DummyVecEnv([lambda: make_env()])
    vec_env_dqn = VecTransposeImage(vec_env_dqn)
    vec_env_dqn = VecFrameStack(vec_env_dqn, 4)
    
    vec_env_ppo = DummyVecEnv([lambda: make_env()])
    vec_env_ppo = VecTransposeImage(vec_env_ppo)
    vec_env_ppo = VecFrameStack(vec_env_ppo, 4)
    
    # Load models
    print("Loading models...")
    timesteps = 500_000
    
    try:
        dqn_model = DQN.load(f"models/DQN/carracing_dqn_{timesteps}.zip", env=vec_env_dqn)
        print("DQN model loaded successfully")
    except Exception as e:
        print(f"Error loading DQN model: {e}")
        return
    
    try:
        ppo_model = PPO.load(f"models/PPO/carracing_ppo_{timesteps}.zip", env=vec_env_ppo)
        print("PPO model loaded successfully")
    except Exception as e:
        print(f"Error loading PPO model: {e}")
        return
    
    # Evaluate models
    print("\n=== Evaluating DQN ===")
    dqn_rewards, dqn_lengths = evaluate_model(dqn_model, vec_env_dqn, num_episodes=50)
    
    print("\n=== Evaluating PPO ===")
    ppo_rewards, ppo_lengths = evaluate_model(ppo_model, vec_env_ppo, num_episodes=50)
    
    # Calculate statistics
    dqn_mean_reward = np.mean(dqn_rewards)
    dqn_std_reward = np.std(dqn_rewards)
    dqn_mean_length = np.mean(dqn_lengths)
    
    ppo_mean_reward = np.mean(ppo_rewards)
    ppo_std_reward = np.std(ppo_rewards)
    ppo_mean_length = np.mean(ppo_lengths)
    
    print(f"\n=== RESULTS ===")
    print(f"DQN  - Mean Reward: {dqn_mean_reward:.2f} ± {dqn_std_reward:.2f}, Mean Length: {dqn_mean_length:.1f}")
    print(f"PPO  - Mean Reward: {ppo_mean_reward:.2f} ± {ppo_std_reward:.2f}, Mean Length: {ppo_mean_length:.1f}")
    
    # Determine winner
    if ppo_mean_reward > dqn_mean_reward:
        print(f"\nPPO wins! (+{ppo_mean_reward - dqn_mean_reward:.2f} reward)")
    else:
        print(f"\nDQN wins! (+{dqn_mean_reward - ppo_mean_reward:.2f} reward)")
    
    # Create plots
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Episode Rewards
        episodes = range(1, 51)
        ax1.plot(episodes, dqn_rewards, 'b-o', label='DQN', alpha=0.7, linewidth=2, markersize=8)
        ax1.plot(episodes, ppo_rewards, 'r-s', label='PPO', alpha=0.7, linewidth=2, markersize=8)
        ax1.axhline(y=dqn_mean_reward, color='b', linestyle='--', alpha=0.5, label=f'DQN Mean: {dqn_mean_reward:.2f}')
        ax1.axhline(y=ppo_mean_reward, color='r', linestyle='--', alpha=0.5, label=f'PPO Mean: {ppo_mean_reward:.2f}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards Comparison (500k timesteps)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2.plot(episodes, dqn_lengths, 'b-o', label='DQN', alpha=0.7, linewidth=2, markersize=8)
        ax2.plot(episodes, ppo_lengths, 'r-s', label='PPO', alpha=0.7, linewidth=2, markersize=8)
        ax2.axhline(y=dqn_mean_length, color='b', linestyle='--', alpha=0.5, label=f'DQN Mean: {dqn_mean_length:.1f}')
        ax2.axhline(y=ppo_mean_length, color='r', linestyle='--', alpha=0.5, label=f'PPO Mean: {ppo_mean_length:.1f}')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length (Steps)')
        ax2.set_title('Episode Lengths Comparison (500k timesteps)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('./resources', exist_ok=True)
        plt.savefig('./resources/model_comparison_500k.png', dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as './resources/model_comparison_500k.png'")
        plt.show()
        
        # Create bar chart comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['DQN', 'PPO']
        mean_rewards = [dqn_mean_reward, ppo_mean_reward]
        std_rewards = [dqn_std_reward, ppo_std_reward]
        
        bars = ax.bar(models, mean_rewards, yerr=std_rewards, capsize=5, 
                      color=['blue', 'red'], alpha=0.7, width=0.6)
        
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title('Model Performance Comparison (500k timesteps)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, mean_rewards, std_rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 5,
                    f'{mean:.2f} ± {std:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('./resources/model_performance_500k.png', dpi=300, bbox_inches='tight')
        print(f"Bar chart saved as './resources/model_performance_500k.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        print("Results summary:")
        print(f"DQN rewards: {dqn_rewards}")
        print(f"PPO rewards: {ppo_rewards}")
    
    # Close environments
    vec_env_dqn.close()
    vec_env_ppo.close()

if __name__ == "__main__":
    main()
