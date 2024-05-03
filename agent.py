import numpy as np
import torch

class QLearning:
    def __init__(self, env, lr=0.01, discount=0.99, epsilon=0.05, grids_per_dim=10):
        # using OpenAI Gym interface
        self.env = env
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.grids_per_dim = grids_per_dim

        self.dim_obs = env.observation_space.shape[0]
        self.dim_act = env.action_space.n
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        self.Q = np.zeros((self.grids_per_dim ** self.dim_obs, env.action_space.n))

    def get_action(self, state):
        state_idx = self.state2idx(state)
        return np.argmax(self.Q[state_idx])

    def update(self, state, action, reward, next_state):
        state_idx = self.state2idx(state)
        next_state_idx = self.state2idx(next_state)
        self.Q[state_idx, action] = (1 - self.lr) * self.Q[state_idx, action] + self.lr * (reward + self.discount * np.max(self.Q[next_state_idx]))

    def get_action_egreedy(self, state):
        state_idx = self.state2idx(state)
        actions = self.Q[state_idx]
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.dim_act)
        else:
            return np.argmax(actions)

    def state2idx(self, state):
        state = 2 * (state - self.low) / (self.high - self.low) - 1 # rescale
        state_digit = np.digitize(state, np.linspace(-1, 1, self.grids_per_dim)) - 1 # digitize
        shape = tuple([self.grids_per_dim] * self.dim_obs)
        idx = np.ravel_multi_index(state_digit, shape)
        return idx


class REINFORCE:
    '''
    Implemement REINFORCE policy gradient using OpenAI Gym interface

    adapted from https://avandekleut.github.io/reinforce/ 
    and https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py
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


