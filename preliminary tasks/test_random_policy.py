"""Test a random policy on the OpenAI Gym Hopper environment.

    
    TASK 1: Play around with this code to get familiar with the
            Hopper environment.

            For example:
                - What is the state space in the Hopper environment? Is it discrete or continuous?
                - What is the action space in the Hopper environment? Is it discrete or continuous?
                - What is the mass value of each link of the Hopper environment, in the source and target variants respectively?
                - what happens if you don't reset the environment even after the episode is over?
                - When exactly is the episode over?
                - What is an action here?
"""
import pdb

import gym

from env.custom_hopper import *

from agent import Policy, Agent



def main():
	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('State space:', env.observation_space) # state-space
	print('Action space:', env.action_space) # action-space
	print('Dynamics parameters:', env.get_parameters()) # masses of each link of the Hopper

	policy = Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
	agent = Agent(policy)

	n_episodes = 500
	render = True

	for episode in range(n_episodes):
		done = False
		state = env.reset()	# Reset environment to initial state
		total_reward = 0
		steps = 0  # To count the steps per episode

		while not done:  # Until the episode is over

			#action = env.action_space.sample()	# Sample random action
		
			#state, reward, done, info = env.step(action)	# Step the simulator to the next timestep
			action, action_log_prob = agent.get_action(state)  # Sample action from policy
			print(f'Action: {action.cpu().numpy()}')  # Debug: Print the action
			next_state, reward, done, info = env.step(action)  # Step the simulator to the next timestep
			agent.store_outcome(state, next_state, action_log_prob, reward, done)  # Store the experience
			print(f'State: {state}, Reward: {reward}, Done: {done}')  # Debug: Print the state, reward, and done flag
			state = next_state
			total_reward += reward
			steps += 1

			if render:
				env.render()

		agent.update_policy()
		#print(f'Episode {episode}, Total Reward: {total_reward}, Policy Loss: {policy_loss}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}')
		print(f'Episode {episode}, Total Reward: {total_reward}, Steps: {steps}')


	

if __name__ == '__main__':
	main()