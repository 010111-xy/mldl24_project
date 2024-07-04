import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from env.custom_hopper import *

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def bootstrapped_discount_rewards(r, gamma, done, next_values):
    bootstrapped_discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        if done[t]:
            running_add = 0
        else:
            running_add = r[t] + gamma * next_values[t]
        bootstrapped_discounted_r[t] = running_add
    return bootstrapped_discounted_r

class Policy(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = nn.Tanh()

        # Actor network
        self.fc1_actor = nn.Linear(state_space, self.hidden)
        self.fc2_actor = nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = nn.Linear(self.hidden, action_space)
        
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        # Critic network
        self.fc1_critic = nn.Linear(state_space, self.hidden)
        self.fc2_critic = nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Actor
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        # Critic
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic(x_critic)
        
        return normal_dist, state_value

class Agent:
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        # Compute discounted returns and bootstrapped estimates
        discounted_returns = discount_rewards(rewards, self.gamma)
        _, next_values = self.policy(next_states)
        bootstrapped_estimates = bootstrapped_discount_rewards(rewards, self.gamma, done, next_values)

        # Compute state values and advantages
        _, state_values = self.policy(states)
        advantages = bootstrapped_estimates - state_values.squeeze()

        # Compute losses
        actor_loss = -torch.sum(action_log_probs * advantages.detach())
        critic_loss = F.mse_loss(state_values.squeeze(), bootstrapped_estimates.detach())
        total_loss = actor_loss + critic_loss

        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist, _ = self.policy(x)

        if evaluation:
            return normal_dist.mean, None
        else:
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

# Training loop
def train(env, agent, num_episodes, max_steps):
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action, action_log_prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            
            agent.store_outcome(state, next_state, action_log_prob, reward, done)
            episode_reward += reward
            state = next_state

            if done:
                break

        agent.update_policy()
        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

    return episode_rewards

# Main execution
if __name__ == "__main__":
    import gym

    env = gym.make('CustomHopper-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim, action_dim)
    agent = Agent(policy)

    num_episodes = 1000
    max_steps = 1000

    episode_rewards = train(env, agent, num_episodes, max_steps)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Save the model
    torch.save(agent.policy.state_dict(), 'custom_hopper_model.pth')

    env.close()