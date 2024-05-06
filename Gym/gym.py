# https://gymnasium.farama.org/content/basic_usage/
# https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63

import gymnasium as gym
from matplotlib import pyplot as plt
from agent import REINFORCE
from policy import LinearPolicy, NNPolicy

import numpy as np
import random
import torch
# import imageio
# from PIL import Image
# import PIL.ImageDraw as ImageDraw

def rescale_states(state, low, high):
    return 2 * (state - low) / (high - low) - 1

########################################################################
# Environment setup
# Uncomment as needed

# # Acrobat
# # https://gymnasium.farama.org/environments/classic_control/acrobot/
# # state: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) theta_dot1 theta_dot2]
# env = gym.make('Acrobot-v1')
# lr = 0.01
# discount = 0.99
# seed = 2
# max_episode_num = 5000


# Mountain Car
# https://gymnasium.farama.org/environments/classic_control/mountain_car/
# state: [position velocity]
env = gym.make('MountainCar-v0', max_episode_steps=1000)
# 0.1: getting small gains but loses it on next iter
# 0,2: many more small gains, don't stick
lr = 1e-4
discount = 0.99
seed = 3
max_episode_num = 2000
########################################################################
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env)

name = env.unwrapped.spec.id
print(name, lr)
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.n
low = env.observation_space.low
high = env.observation_space.high

policy = LinearPolicy(obs_space_dims, action_space_dims) # for Acrobat-V1

random.seed(seed)
np.random.seed(seed)
agent = REINFORCE(env, policy, discount=discount, lr = lr)

episode_rewards = []
episode_lengths = []

max_episode_num = 2000
for episode in range(max_episode_num):
    state, info = wrapped_env.reset(seed=seed)
    done = False

    actions = []
    states = []
    t = 0
    while not done:
        state = rescale_states(state, low, high)
        if name == 'MountainCar-v0': 
            if t % 5 == 0:
                action = agent.sample_action(state)
        else:
            action = agent.sample_action(state)

        # print(episode, t, action)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)

        # if name == 'MountainCar-v0':
        #     # give a small reward for climbing out of the sink
        #     reward += np.abs(obs[0]) * 0.5 

        agent.rewards.append(reward) # store the reward

        done = terminated or truncated
        actions.append(action)
        states.append(state)
        state = obs
        t += 1
        
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


# # render
# env_render = gym.make(name, render_mode='rgb_array')
# done = False
# frames = []
# state, info = env_render.reset(seed=seed)
# while not done:
#     action = agent.sample_action(rescale_states(state, low, high))
#     frame = env.render(mode='rgb_array')
#     im = Image.fromarray(frame)
#     drawer = ImageDraw.Draw(im)
#     state, reward, done, _ = env_render.step(action)

#     frames.append(frame)

# imageio.mimwrite(f'videos/{name}_agent.gif', frames, fps=60)