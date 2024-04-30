import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

gamma = 0.99

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 2) # Adjust as per your env

    def forward(self, x):
        return torch.softmax(self.l1(x), dim=-1)

def reinforce(policy, optimizer, episodes):
    episode_rewards = []
    for episode in range(episodes):
        state,_ = env.reset()
        done = False
        log_probs = []
        rewards = []
        while not done:
            state = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy(state)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            log_probs.append(m.log_prob(action))
            rewards.append(reward)

        episode_rewards.append(np.mean(rewards[-10:]))
        optimizer.zero_grad()
        policy_loss = calculate_policy_loss(log_probs, rewards)
        policy_loss.backward()
        optimizer.step()

    return episode_rewards

def calculate_policy_loss(log_probs, rewards):
    discounts = [gamma**i for i in range(len(rewards))]
    R = sum([a*b for a,b in zip(discounts, rewards)])
    policy_loss = -sum([log_prob * R for log_prob in log_probs])
    return policy_loss

env = gym.make("CartPole-v1")
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)
rewards = reinforce(policy, optimizer, 1000)

plt.plot(range(1000), rewards)
plt.savefig("plots/rewards.png")