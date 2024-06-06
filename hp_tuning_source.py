import gym
from env.custom_hopper import *
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Define the environment name
env_name = gym.make('CustomHopper-source-v0')

# Define the function to evaluate the agent
def evaluate_agent(agent, env, num_episodes=50):
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)

# Initialize the environment
env = DummyVecEnv([lambda: env_name])

# Define the hyperparameters to tune
learning_rates = [0.001, 0.01, 0.1]
n_steps_values = [128, 256, 512]
n_epochs_values = [3, 5, 10]
clip_range_values = [0.1, 0.2, 0.3]

best_mean_reward = -np.inf
best_hyperparameters = {}

# Grid search
for lr in learning_rates:
    for n_steps in n_steps_values:
        for n_epochs in n_epochs_values:
            for clip_range in clip_range_values:
                # Define hyperparameters for training on the source domain
                hyperparameters = {
                    'learning_rate': lr,
                    'n_steps': n_steps,
                    'batch_size': 64,
                    'gamma': 0.99,
                    'ent_coef': 0.01,
                    'clip_range': clip_range,
                    'n_epochs': n_epochs,
                    'max_grad_norm': 0.5,
                    'vf_coef': 0.5,
                    'tensorboard_log': None
                }

                # Train the agent
                agent = PPO('MlpPolicy', env, verbose=0, **hyperparameters)
                agent.learn(total_timesteps=100000)

                # Evaluate the agent
                mean_reward = evaluate_agent(agent, env)

                # Print the mean reward and hyperparameters
                print(f"Mean reward: {mean_reward:.2f}, Hyperparameters: {hyperparameters}")

                # Update the best hyperparameters if needed
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    best_hyperparameters = hyperparameters

print("Best mean reward:", best_mean_reward)
print("Best hyperparameters:", best_hyperparameters)
