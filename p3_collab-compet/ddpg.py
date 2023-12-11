import numpy as np
import random
import copy
from collections import namedtuple, deque

from models import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

LR_ACTOR = 1.1e-4         # learning rate of the actor 
LR_CRITIC = 3.3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

NOISE_DECAY = 0.99
BEGIN_TRAINING_AT = 500
NOISE_START = 1.0
NOISE_END = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, observation_size, action_size, idx_agent, n_agents):

        self.observation_size = observation_size
        self.action_size = action_size
        self.idx_agent = idx_agent = torch.tensor([idx_agent]).to(device)
        print(self.idx_agent)

        # actor (network from observation to action) 
        self.actor_local = Actor(observation_size, action_size).to(device)
        self.actor_target = Actor(observation_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # critic (network from all agent observations, all acttions to action value) 
        self.critic_local = Critic(n_agents*observation_size, n_agents*action_size).to(device)
        self.critic_target = Critic(n_agents*observation_size, n_agents*action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC,
                                           weight_decay=WEIGHT_DECAY)

        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)
        
        self.noise = RandomNoise(self.action_size,
                                 NOISE_START, NOISE_END, NOISE_DECAY,
                                 BEGIN_TRAINING_AT, 1)
    
    def get_index(self):
        return self.idx_agent

    def act(self, observation, i_episode=0, add_noise=True):

        state = torch.from_numpy(observation).float().to(device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample(i_episode)

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, actions_target, actions_pred):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        observations, actions, rewards, next_states, dones = experiences
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        
        # ---------------------------- update critic ----------------------------
        actions_target = torch.cat(actions_target, dim=1).to(device)
        
        Q_targets_next = self.critic_target(next_states.reshape(next_states.shape[0], -1), actions_target.reshape(next_states.shape[0], -1))
        Q_targets = rewards.index_select(1, self.idx_agent).squeeze(1) + (gamma * Q_targets_next * (1 - dones.index_select(1, self.idx_agent).squeeze(1)))

        Q_expected = self.critic_local(observations.reshape(observations.shape[0], -1), actions.reshape(actions.shape[0], -1))
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        actor_loss = -self.critic_local(observations.reshape(observations.shape[0], -1), actions_pred.reshape(actions_pred.shape[0], -1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()                      

        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau*local_param.data +
                                    (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
    
class RandomNoise:
    """Random noise process."""
    def __init__(self, size, weight, min_weight, noise_decay,
                 begin_noise_at, seed):
        self.size = size
        self.weight_start = weight
        self.weight = weight
        self.min_weight = min_weight
        self.noise_decay = noise_decay
        self.begin_noise_at = begin_noise_at
        self.seed = random.seed(seed)

    def reset(self):
        self.weight = self.weight_start

    def sample(self, i_episode):
        pwr = max(0, i_episode - self.begin_noise_at)
        if pwr > 0:
            self.weight = max(self.min_weight, self.noise_decay**pwr)
        return self.weight * 0.5 * np.random.standard_normal(self.size)