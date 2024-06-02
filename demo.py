import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_network(state)
        action = np.random.choice(len(probs[0]), p=probs.detach().numpy()[0])
        return action

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = sum([self.gamma**i * rewards[t+i] for i in range(len(rewards)-t)])
            discounted_rewards.append(Gt)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        loss = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * Gt)
        loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ActorCritic:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.value_network = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_network(state)
        action = np.random.choice(len(probs[0]), p=probs.detach().numpy()[0])
        log_prob = torch.log(probs.squeeze(0)[action])
        return action, log_prob

    def update_policy(self, log_probs, states, rewards):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = sum([self.gamma**i * rewards[t+i] for i in range(len(rewards)-t)])
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        states = torch.FloatTensor(states)
        values = self.value_network(states).squeeze()

        advantages = discounted_rewards - values

        policy_loss = []
        value_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
            value_loss.append(nn.functional.mse_loss(values, discounted_rewards))

        self.policy_optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss = torch.stack(value_loss).sum()
        value_loss.backward()
        self.value_optimizer.step()

def train(agent, env, episodes, max_timesteps, update_method):
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        states = []

        for t in range(max_timesteps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            states.append(state)

            state = next_state

            if done:
                break

        if update_method == 'REINFORCE':
            agent.update_policy(rewards, log_probs)
        elif update_method == 'ActorCritic':
            agent.update_policy(log_probs, states, rewards)

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards)}")

if __name__ == "__main__":
    env = gym.make('Hopper-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent_type = 'REINFORCE'  # 'REINFORCE' or 'ActorCritic'
    episodes = 1000
    max_timesteps = 200
    lr = 1e-3
    gamma = 0.99

    if agent_type == 'REINFORCE':
        agent = REINFORCE(state_dim, action_dim, lr, gamma)
    elif agent_type == 'ActorCritic':
        agent = ActorCritic(state_dim, action_dim, lr, gamma)

    train(agent, env, episodes, max_timesteps, agent_type)

