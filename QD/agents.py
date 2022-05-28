import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class QD_agent:

    def __init__(self, env):
        self.action_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.model = np.zeros((self.action_dim, self.obs_dim))

    def select_action(self, state):
        return self.model @ state

    def update_params(self,new_model):
        self.model = new_model.reshape((self.action_dim, self.obs_dim))

    def flatten(self):
        return self.model.flatten()

class QD_agent_NN:

    def __init__(self, env, ls = 32):
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        l1 = ls; l2 = ls; l3 = l2

        # Construct Hidden Layer 1
        self.w_l1 = nn.Linear(obs_dim, l1)
        self.lnorm1 = nn.LayerNorm(l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        self.lnorm1 = nn.LayerNorm(l1)

        # Out
        self.w_out = nn.Linear(l3, action_dim)

        # Init

        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)
 
    def forward(self, input):

        # Hidden Layer 1
        out = self.w_l1(input)
        if self.args.use_ln: out = self.lnorm1(out)
        out = out.tanh()

        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = out.tanh()

        # Out
        out = (self.w_out(out)).tanh()

        return out

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        return self.forward(state).cpu().data.numpy().flatten()