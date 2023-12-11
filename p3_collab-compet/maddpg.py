# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import numpy as np
from utilities import soft_update, transpose_to_tensor, transpose_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.996           # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 3        # Udpate every
NB_LEARN = 4

class MADDPG:
    def __init__(self, observation_size, action_size, n_agents, discount_factor=0.95, tau=0.02):

        self.observation_size = observation_size
        self.action_size = action_size
        self.n_agents = n_agents;
        self.discount_factor = discount_factor
        self.tau = tau

        self.agent_list = [DDPGAgent(observation_size, action_size, idx_agent, n_agents) for idx_agent in range(n_agents)]

        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        
    def step(self, states, actions, rewards, next_states, dones, t):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.replay_buffer.add(states, actions, rewards, next_states, dones)
            

        if len(self.replay_buffer) > BATCH_SIZE and (t % UPDATE_EVERY) == 0:
            for _ in range(NB_LEARN):
                for agent in self.agent_list:
                    experiences = self.replay_buffer.sample()
                    self.learn(experiences, agent, GAMMA)
                    
                for agent in self.agent_list:
                    agent.soft_update(agent.critic_local,
                          agent.critic_target,
                          TAU)
                    agent.soft_update(agent.actor_local,
                          agent.actor_target,
                          TAU)    
              
    def learn(self, experiences, agent, gamma):
        states, actions, _, _, _ = experiences

        actions_target =[agent_j.actor_target(states.index_select(1, torch.tensor([j]).to(device)).squeeze(1)) 
                         for j, agent_j in enumerate(self.agent_list)]
        
        agent_action_pred = agent.actor_local(states.index_select(1, agent.get_index()).squeeze(1))
        actions_pred = [agent_action_pred if j==agent.get_index().numpy()[0] else actions.index_select(1, torch.tensor([j]).to(device)).squeeze(1) 
                        for j, agent_j in enumerate(self.agent_list)]
        
        agent.learn(experiences,
                    gamma,
                    actions_target,
                    actions_pred)


    def act(self, states, i_episode=0, add_noise=True):
        actions = [np.squeeze(agent.act(np.expand_dims(state, axis=0), i_episode, add_noise), axis=0) for agent, state in zip(self.agent_list, states)]
        return np.stack(actions)
       
        
    def reset(self):
        for agent in self.agent_list:
            agent.reset()


    ######  

            
from collections import namedtuple, deque
import random

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)