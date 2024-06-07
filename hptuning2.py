rimport gym
from env.custom_hopper import *
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna

# Define the environment name
source_env_name = 'CustomHopper-source-v0'
target_env_name = 'CustomHopper-target-v0'

# Define the function to evaluate the agent
def evaluate_agent(agent, env, num_episodes=50, render=False):
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()
            action, _ = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)

# Initialize the environments
source_env = DummyVecEnv([lambda: gym.make(source_env_name)])
target_env = DummyVecEnv([lambda: gym.make(target_env_name)])

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-4, 1e-2)
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
    n_epochs = trial.suggest_int('n_epochs', 1, 10)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.4)

    # Define hyperparameters for training on the source domain
    hyperparameters = {
        'learning_rate': learning_rate,
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
    agent = PPO('MlpPolicy', source_env, verbose=0, **hyperparameters)
    agent.learn(total_timesteps=100000)

    # Evaluate the agent
    mean_reward = evaluate_agent(agent, source_env)
    
    return mean_reward

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)

# Get the best hyperparameters
best_hyperparameters = study.best_params
best_mean_reward = study.best_value

print("Best mean reward:", best_mean_reward)
print("Best hyperparameters:", best_hyperparameters)

# Train the source agent with the best hyperparameters
source_agent = PPO('MlpPolicy', source_env, verbose=1, **best_hyperparameters)
source_agent.learn(total_timesteps=100000)

# Train the target agent with default hyperparameters
target_agent = PPO('MlpPolicy', target_env, verbose=1)
target_agent.learn(total_timesteps=100000)

# Evaluate the agents with rendering enabled
source_source_return = evaluate_agent(source_agent, source_env, render=True)
source_target_return = evaluate_agent(source_agent, target_env, render=True)
target_target_return = evaluate_agent(target_agent, target_env, render=True)

# Close the environments after evaluation
source_env.close()
target_env.close()

# Print results
print("Source→Source return:", source_source_return)
print("Source→Target return (lower bound):", source_target_return)
print("Target→Target return (upper bound):", target_target_return)
