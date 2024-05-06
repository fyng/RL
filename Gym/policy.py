# adapted from https://gibberblot.github.io/rl-notes/single-agent/policy-gradients.html


# https://users.ece.cmu.edu/~yuejiec/ece18813B_notes/lecture16-linear-function-approximation.pdf

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

class LinearPolicy(nn.Module):
    def __init__(self, obs_space_dims, action_space_dims):
        super().__init__()
        # Linear policy
        self.model = nn.Linear(obs_space_dims, action_space_dims)

    def forward(self, x):
        x = torch.as_tensor(x)
        p = self.model(x)
        pi = Categorical(logits=p)
        return pi

class NNPolicy(nn.Module):
    def __init__(self, obs_space_dims, action_space_dims):
        super().__init__()
        # NN policy
        hidden_space1 = 6
        self.model = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, action_space_dims),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = torch.as_tensor(x)
        p = self.model(x)
        pi = Categorical(logits=p)
        return pi

