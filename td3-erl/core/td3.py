import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam

from core import replay_memory
from core.genetic_agent import Actor
from core.mod_utils import hard_update, soft_update, LayerNorm

from typing import Tuple, Dict, List

MAX_GRAD_NORM = 1

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        # layer sizes
        # l1 = 200; l2 = 300; l3 = l2    # original PDERL values (no tuning done)
        l1 = 64; l2 = 64;            # NOTE these worked for TD3-only control 

        # Critic 1
        self.bnorm_1 = nn.BatchNorm1d(args.state_dim + args.action_dim)  # batch norm
        self.l1_1 = nn.Linear(args.state_dim + args.action_dim, l1)
        self.lnorm1_1 = LayerNorm(l1)
        self.l2_1 = nn.Linear(l1, l2)
        self.lnorm2_1 = LayerNorm(l2)
        self.lout_1 = nn.Linear(l2, 1)

        # Critic 2
        self.bnorm_2 = nn.BatchNorm1d(args.state_dim + args.action_dim)  # batch norm
        self.l1_2 = nn.Linear(args.state_dim + args.action_dim, l1)
        self.lnorm1_2 = LayerNorm(l1)
        self.l2_2 = nn.Linear(l1, l2)
        self.lnorm2_2 = LayerNorm(l2)
        self.lout_2 = nn.Linear(l2, 1)

        # Initlaise wights with smaller values
        self.lout_1.weight.data.mul_(0.1);self.lout_1.bias.data.mul_(0.1)
        self.lout_2.weight.data.mul_(0.1);self.lout_2.bias.data.mul_(0.1)


        self.to(self.args.device)

    def forward(self, state, action):
        # ------ Critic 1 ---------
        input = torch.cat((state,action), 1)
        input = self.bnorm_1(input)

        # hidden Layer 1 (Input Interface)
        out = self.l1_1(input)
        out = self.lnorm1_1(out)
        out = F.elu(out)

        # hidden Layer 2
        out = self.l2_1(out)
        out = self.lnorm2_1(out)
        out = F.elu(out)

        # output interface
        out1 = self.lout_1(out)

        # ------ Critic 2 ---------
        # hidden Layer 1 (Input Interface)
        input = torch.cat((state,action), 1)
        input = self.bnorm_2(input)

        out = self.l1_2(input)
        out = self.lnorm1_2(out)
        out = F.elu(out)

        # hidden Layer 2
        out = self.l2_2(out)
        out = self.lnorm2_2(out)
        out = F.elu(out)

        # output interface
        out2 = self.lout_2(out)

        return out1, out2


class TD3(object):
    def __init__(self, args):

        self.args = args
        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)

        # Initialize actor
        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr = self.args.lr)

        # Initialise critics
        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.args.lr)

        # Initliase loss
        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        # Make sure target is with the same weights
        hard_update(self.actor_target, self.actor)  
        hard_update(self.critic_target, self.critic)


    def update_parameters(self, batch, iteration : int, champion_policy = None) -> Tuple[float,float]:
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch
        pgl = None
        with torch.no_grad():
            # Load everything to GPU if not already
            self.actor_target.to(self.args.device)
            self.critic_target.to(self.args.device)
            self.critic.to(self.args.device)
            state_batch = state_batch.to(self.args.device)
            next_state_batch = next_state_batch.to(self.args.device)
            action_batch = action_batch.to(self.args.device)
            reward_batch = reward_batch.to(self.args.device)
            done_batch = done_batch.to(self.args.device)

            # Select action according to policy 
            next_action_batch = self.actor_target.forward(next_state_batch)

            # Compute the target Q values
            target_Q1, target_Q2 = self.critic_target.forward(next_state_batch, next_action_batch)
            next_Q = torch.min(target_Q1, target_Q2)
            next_Q = next_Q * (1 - done_batch) # Done mask
            target_q = reward_batch + (self.gamma * next_Q).detach()

        # Get current Q estimates
        current_q1, current_q2 = self.critic.forward(state_batch, action_batch)

        # Compute critic losses
        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)
        TD =  loss_q1 + loss_q2

        # Optimize the criticss
        self.critic_optim.zero_grad()
        TD.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
        self.critic_optim.step()

        # Actor Update
        if iteration % self.args.policy_update_freq == 0:
            self.actor_optim.zero_grad()

            # retrieve value of the critics
            est_q1,_ = self.critic.forward(state_batch, self.actor.forward(state_batch))
            policy_grad_loss = -torch.mean(est_q1)             # add minus to make it a loss

            # backprop
            policy_grad_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.actor_optim.step()

            # smooth target updates 
            if champion_policy is not None:
                soft_update(self.actor_target, champion_policy, self.tau)
            else:
                soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

            pgl = policy_grad_loss.data.cpu().numpy()
        return pgl, TD.data.cpu().numpy()


