"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy
from agent_reinforce import REINFORCE, PolicyNetwork

def parse_args():
		# actor-critic train records
		# model0.mdl discount_rewards mse_loss
		# model1.mdl discount_rewards l2_loss
		# model2.mdl bootstrapped_discount_rewards mse_loss
		# model3.mdl bootstrapped_discount_rewards l2_loss
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='actor-critic', type=str, help='model type [REINFORCE, actor-critic]')
    parser.add_argument('--model', default='model0.mdl', type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=500, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	if args.model_type == 'REINFORCE':
		policy = PolicyNetwork(observation_space_dim, action_space_dim)
		agent = REINFORCE(policy, device=args.device)
	else:
		policy = Policy(observation_space_dim, action_space_dim)
		agent = Agent(policy, device=args.device)
	
	policy.load_state_dict(torch.load(args.model), strict=True)

	

	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward

		print(f"Episode: {episode} | Return: {test_reward}")
	

if __name__ == '__main__':
	main()