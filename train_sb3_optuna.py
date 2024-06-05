"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
import optuna

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn


def test():
    model = PPO.load('model_ppo_best_params_optuna')
    env = gym.make('CustomHopper-source-v0')
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()

def tuning(trial):

    n_steps = trial.suggest_int('n_steps', 2048, 8192)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    # learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    n_epochs = trial.suggest_int('n_epochs', 1, 10)
    initial_lr = trial.suggest_float('initial_lr', 1e-5, 1e-2, log=True)
    final_lr = trial.suggest_float('final_lr', 1e-6, initial_lr, log=True)
    lr_steps = trial.suggest_int('lr_steps', 50000, 200000)
    learning_rate = get_linear_fn(initial_lr, final_lr, lr_steps)
    
    env = make_vec_env(lambda: gym.make('CustomHopper-source-v0'), n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        #batch_size=batch_size,
        verbose=0 # close log output
    )
    model.learn(total_timesteps=100000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward

def train_best_params():
    study = optuna.create_study(direction='maximize')
    study.optimize(tuning, n_trials=200)
    print("Best params: ", study.best_params)
    train_with_params(study.best_params)


def train_with_params(best_params):
    learning_rate = get_linear_fn(best_params['initial_lr'], best_params['final_lr'], best_params['lr_steps'])

    env = make_vec_env('CustomHopper-source-v0', n_envs=4)
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=learning_rate,
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        ent_coef=best_params['ent_coef'],
        clip_range=best_params['clip_range'],
        n_epochs=best_params['n_epochs'],
        verbose=1
    )
    
    model.learn(total_timesteps=2000000)
    model.save("model_ppo_best_params_optuna")

if __name__ == '__main__':
    #test()
    train_best_params()