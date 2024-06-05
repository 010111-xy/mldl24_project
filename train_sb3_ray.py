"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym

from ray import tune
from ray.air import session
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from ray.tune.schedulers import FIFOScheduler
from register_envs import register_custom_hopper

def test():
    model = PPO.load('model_ppo_best_params_ray')
    env = gym.make('CustomHopper-source-v0')
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()

config = {
    'learning_rate': tune.loguniform(1e-5, 1e-2),
    'n_steps': tune.choice([2048, 4096, 8192]),
    'gamma': tune.uniform(0.9, 0.9999),
    'ent_coef': tune.uniform(0.0, 0.1),
    'clip_range': tune.uniform(0.1, 0.4),
    'n_epochs': tune.choice([1, 5, 10]),
    'batch_size': tune.choice([64, 128, 256, 512, 1024])
}

def tuning(config):
    register_custom_hopper()
    env = make_vec_env('CustomHopper-source-v0', n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=config['n_steps'],
        gamma=config['gamma'],
        learning_rate=config['learning_rate'],
        ent_coef=config['ent_coef'],
        clip_range=config['clip_range'],
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        verbose=0 # close log output
    )
    model.learn(total_timesteps=100000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    session.report({'mean_reward': mean_reward})

def train_best_params():
    
    analysis = tune.run(
        tuning,
        config=config,
        num_samples=200,
        scheduler= FIFOScheduler(),
        max_concurrent_trials=4 # OOM
    )
    train_with_params(analysis.best_config)


def train_with_params(best_params):
    env = make_vec_env('CustomHopper-source-v0', n_envs=4)
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=best_params['learning_rate'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        ent_coef=best_params['ent_coef'],
        clip_range=best_params['clip_range'],
        n_epochs=best_params['n_epochs'],
        batch_size=best_params['batch_size'],
        verbose=1
    )
    
    model.learn(total_timesteps=1000000)
    model.save("model_ppo_best_params_ray")

if __name__ == '__main__':
    #test()
    train_best_params()