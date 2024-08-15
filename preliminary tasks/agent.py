import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


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


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden) #torch.nn.Linear(dominio, codominio)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1) #we put 1 because critic returns a scalar value, which is V(s)
        

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic(x_critic)

        
        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.critic = policy.to(self.train_device)  # Use policy as critic
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = [] #done has boolean values, it has True if the state we are in is terminal


    def update_policy(self):
        #CONVERT LIST OF EXPERIENCES INTO TENSORS ---> torch.stack(..)
        #AND MOVE THEM IN THE TRAINING DEVICE
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)


        #
        # TASK 2:
        #   - compute discounted returns
        discounted_returns = discount_rewards(rewards, self.gamma)
        # compute policy gradient loss function given actions and returns
        policy_loss = -torch.sum(discounted_returns.detach()*action_log_probs)
         # - compute gradients and step the optimizer
        self.optimizer.zero_grad()     # Zero the gradients of the optimizer
        policy_loss.backward()         # Backpropagate the policy loss
        self.optimizer.step()          # Step the optimizer to update the policy parameters


        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        _, next_values = self.policy(next_states)
        bootstrapped_estimates = bootstrapped_discount_rewards(rewards, self.gamma, done, next_values)
        #  - compute advantage 
        #V(s_t)
        _, state_values = self.policy(states)
        #state_values = self.critic(states)
        advantages = bootstrapped_estimates-state_values
        #   - compute actor loss and critic loss
        actor_loss = -torch.sum(action_log_probs * advantages.detach())
        critic_loss = F.mse_loss(state_values.squeeze(), bootstrapped_estimates.detach)
        #   - compute gradients and step the optimizer
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        #

        #clear buffers when you reach the end of the episode
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []
        return 


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
