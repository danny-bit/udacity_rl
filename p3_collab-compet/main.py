from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from maddpg import MADDPG
import torch

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

seeding()

env = UnityEnvironment(file_name="p3_collab-compet/Tennis_Windows_x86_64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size

# examine the state space 
states = env_info.vector_observations
observation_size = states.shape[1]

print('##   ')
print('Number of agents:', num_agents)
print('Size of each the actionspace of each agent:', action_size)
print('Size of observations space of each agent:', observation_size)
print('Full state size:', states.shape[0], 'x', observation_size)

## 
maddpg = MADDPG(observation_size=observation_size, action_size=action_size, n_agents=num_agents)

##
def ddpg(n_episodes=6000, print_every=100):
    f = open('log_training_p3coab.csv','w')

    scores_deque = deque(maxlen=print_every)
    peak_score = 0.0
    scores = []
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  
        maddpg.reset()
        score = np.zeros(num_agents)
        t = 0
        while True:
            t = t + 1
            actions = maddpg.act(states, i_episode, add_noise=True)
            env_info = env.step(actions)[brain_name]   
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            maddpg.step(states, actions, rewards, next_states, dones, t)
            
            states = next_states
            score += rewards
            if any(dones):
                break 
                
        scores_deque.append(np.max(score))
        scores.append(np.max(score))

        last_peak_score = peak_score
        peak_score = float(np.max([np.mean(scores_deque), peak_score]))

        f.write("%.4f, %.4f, %.4f, %.4f\n" % (score[0],score[1],np.max(score),np.mean(scores_deque)))

        if (i_episode % 10) == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque) >= 0.5:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - print_every, np.mean(scores_deque)))
            if (peak_score > last_peak_score):
                for i, agent in enumerate(maddpg.agent_list):
                    torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_best{i}.pth')
                    torch.save(agent.critic_local.state_dict(), f'checkpoint_critic_best{i}.pth')
        
        f.flush()
    
    f.close()         
    return scores

scores = ddpg()

## 
env.close()