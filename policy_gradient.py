# adapted from 
# https://avandekleut.github.io/reinforce/ 
# and https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py

import numpy as np
import torch

class REINFORCE:
    '''
    Implemement REINFORCE policy gradient using OpenAI Gym interface
    '''
    def __init__(self, env, policy, discount=0.99, lr=0.01) -> None:
        self.lr = lr
        self.gamma = discount

        self.log_probs = []
        self.rewards = []

        self.env = env
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def sample_action(self, state):
        '''
        Return an discrete action using pi(a|s) 
        '''
        a_t = self.policy(state).sample().item()
        return a_t


    def update(self, states, actions):
        states = torch.tensor(np.array(states))
        actions = torch.tensor(np.array(actions))

        running_g = 0
        gs = []

        # discounted return (backwards) Z
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        # baseline = running average of rewards
        baseline = np.convolve(self.rewards, np.ones(10)/10)[:-9]
        deltas = torch.tensor(gs - baseline)
        log_probs = self.policy(states).log_prob(actions)

        # loss update on the linear model
        # loss = 0
        # minimize -1 * prob * reward obtained        
        # for log_prob, delta in zip(self.log_probs, deltas):
        #     loss = log_prob * delta * (-1)

        loss = torch.mean(-log_probs * deltas)
    
            # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


