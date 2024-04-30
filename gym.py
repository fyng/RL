# https://gymnasium.farama.org/content/basic_usage/
# https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63

import gymnasium as gym
from matplotlib import pyplot as plt
from policy_gradient import REINFORCE
from policy import LinearPolicy

import numpy as np
import random
import torch

def rescale_states(state, low, high):
    return 2 * (state - low) / (high - low) - 1

########################################################################
# Uncomment as needed

# # Acrobat
# # https://gymnasium.farama.org/environments/classic_control/acrobot/
# # state: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) theta_dot1 theta_dot2]
# env = gym.make('Acrobot-v1')
# lr = 0.01
# discount = 0.99


# Mountain Car
# https://gymnasium.farama.org/environments/classic_control/mountain_car/
# state: [position velocity]
env = gym.make('MountainCar-v0')
env._max_episode_steps = 1000
lr = 0.01
discount = 0.99
########################################################################
name = env.unwrapped.spec.id
print(name)

obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.n
low = env.observation_space.low
high = env.observation_space.high
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env)

policy = LinearPolicy(obs_space_dims, action_space_dims)

print(action_space_dims)

# for seed in [1,2,3,5,8]:
seed = 2
random.seed(seed)
np.random.seed(seed)
agent = REINFORCE(env, policy, discount=discount, lr = lr)

episode_rewards = []
episode_lengths = []

max_episode_num = 5000


for episode in range(max_episode_num):
    state, info = wrapped_env.reset()
    done = False

    actions = []
    states = []
    while not done:
        action = agent.sample_action(rescale_states(state, low, high))
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        agent.rewards.append(reward) # store the reward

        done = terminated or truncated
        actions.append(action)
        states.append(state)

        state = obs
    
    agent.update(states, actions)

    # calculate the discounted future rewards
    episode_rewards.append(wrapped_env.return_queue[-1])
    episode_lengths.append(wrapped_env.length_queue[-1])

    if episode % 100 == 0:
        avg_reward = np.mean(wrapped_env.return_queue)
        avg_length = np.mean(wrapped_env.length_queue)
        print(f'Episode {episode}, average reward: {avg_reward:.2f}, average length: {avg_length:.2f}')

# plots
f1, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(range(0, max_episode_num), episode_lengths)
axes[0].set_title('Average steps to goal in an episode')

axes[1].plot(range(0, max_episode_num), episode_rewards)
axes[1].set_title('Average reward in an episode')
plt.tight_layout()
plt.savefig(f'plots/REINFORCE_{name}.png')