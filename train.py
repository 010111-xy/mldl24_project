"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from datetime import datetime
from env.custom_hopper import *
from agent import Agent, Policy
from agent_reinforce import REINFORCE, PolicyNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--model-type', default='actor-critic', type=str, help='model type [REINFORCE, actor-critic]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	print('Model type', args.model_type)

	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	if args.model_type == 'actor-critic':
		policy = Policy(observation_space_dim, action_space_dim)
		agent = Agent(policy, device=args.device)
		print('----actor-critic---')
	else:
		policy = PolicyNetwork(observation_space_dim, action_space_dim)
		agent = REINFORCE(policy, device=args.device)
		print('----REINFORCE---')

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		# update the policy
		agent.update_policy()
		
		if (episode+1)%args.print_every == 0:
			print('Time:', datetime.now())
			print('Training episode:', episode + 1)
			print('Episode return:', train_reward)


	torch.save(agent.policy.state_dict(), "model.mdl")

	

if __name__ == '__main__':
	main()