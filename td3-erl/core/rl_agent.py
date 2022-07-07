import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from core import replay_memory

from core.genetic_agent import Actor
from core.mod_utils import hard_update, soft_update, LayerNorm

from typing import Tuple, Dict, List

class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        # layer sizes
        l1 = 200; l2 = 300; l3 = l2

        # Critic 1
        self.l1_1 = nn.Linear(args.state_dim + args.action_dim, l1)
        self.lnorm1_1 = LayerNorm(l1)
        self.l2_1 = nn.Linear(l1, l2)
        self.lnorm2_1 = LayerNorm(l2)
        self.out_1 = nn.Linear(l3, 1)

        # Critic 2
        self.l1_2 = nn.Linear(args.state_dim + args.action_dim, l1)
        self.lnorm1_2 = LayerNorm(l1)
        self.l2_2 = nn.Linear(l1, l2)
        self.lnorm2_2 = LayerNorm(l2)
        self.out_2 = nn.Linear(l3, 1)

        # Initlaise wights with smaller values
        self.out_1.weight.data.mul_(0.1);self.out_1.bias.data.mul_(0.1)
        self.out_2.weight.data.mul_(0.1);self.out_2.bias.data.mul_(0.1)


        self.to(self.args.device)

    def forward(self, state, action):
        # ------ Critic 1 ---------
        # hidden Layer 1 (Input Interface)
        input = torch.cat((state,action), 1)
        out = F.elu(self.l1_1(input))
        out = self.lnorm1_1(out)

        # hidden Layer 2
        out = self.l2_1(out)
        out = self.lnorm2_1(out)
        out = F.elu(out)

        # output interface
        out1 = self.out_1(out)

        # ------ Critic 2 ---------
        # hidden Layer 1 (Input Interface)
        input = torch.cat((state,action), 1)
        out = F.elu(self.l1_2(input))
        out = self.lnorm1_2(out)

        # hidden Layer 2
        out = self.l2_2(out)
        out = self.lnorm2_2(out)
        out = F.elu(out)

        # output interface
        out2 = self.out_2(out)

        return out1, out2


class TD3(object):
    def __init__(self, args):

        self.args = args
        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)

        # Initialize actor
        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.5e-3)

        # Initialise critics
        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)


        # Initliase loss
        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        # Make sure target is with the same weights
        hard_update(self.actor_target, self.actor)  
        hard_update(self.critic_target, self.critic)


    def update_parameters(self, batch, iteration : int) -> Tuple[float,float]:
        pgl = None
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

        with torch.no_grad():
            # Load everything to GPU if not already
            self.actor_target.to(self.args.device)
            self.critic_target.to(self.args.device)
            self.critic.to(self.args.device)
            state_batch = state_batch.to(self.args.device)
            next_state_batch = next_state_batch.to(self.args.device)
            action_batch = action_batch.to(self.args.device)
            reward_batch = reward_batch.to(self.args.device)
            if self.args.use_done_mask: done_batch = done_batch.to(self.args.device)

            # Select action according to policy 
            next_action_batch = self.actor_target.forward(next_state_batch)

            # Compute the target Q values
            target_Q1, target_Q2 = self.critic_target.forward(next_state_batch, next_action_batch)
            next_Q = torch.min(target_Q1, target_Q2)
            if self.args.use_done_mask: next_Q = next_Q * (1 - done_batch) # Done mask
            target_q = reward_batch + (self.gamma * next_Q).detach()

        # Get current Q estimates
        current_q1, current_q2 = self.critic.forward(state_batch, action_batch)

        # Compute critic loss
        TD = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optim.zero_grad()
        TD.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        # Actor Update
        if iteration % self.args.policy_update_freq ==0:
            self.actor_optim.zero_grad()
            # retrieve value of the first critic
            est_q1, _ = self.critic.forward(state_batch, self.actor.forward(state_batch))
            policy_grad_loss = -(est_q1).mean()   # add minus to make it a loss
            policy_loss = policy_grad_loss

            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_optim.step()

            # I think pytorch might complain about netwok 'version' 
            # if the soft upadtes are before grad steps
            # NOTE check this!!
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

            pgl = policy_grad_loss.data.cpu().numpy()

        return pgl, TD.data.cpu().numpy()



class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        l1 = 200; l2 = 300; l3 = l2

        # Construct input interface (Hidden Layer 1)
        self.w_state_l1 = nn.Linear(args.state_dim, l1)
        self.w_action_l1 = nn.Linear(args.action_dim, l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(2*l1, l2)
        self.lnorm2 = LayerNorm(l2)

        # Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input, action):

        # Hidden Layer 1 (Input Interface)
        out_state = F.elu(self.w_state_l1(input))
        out_action = F.elu(self.w_action_l1(action))
        out = torch.cat((out_state, out_action), 1)

        # Hidden Layer 2
        out = self.w_l2(out)
        out = self.lnorm2(out)
        out = F.elu(out)

        # Output interface
        out = self.w_out(out)

        return out


class DDPG(object):
    def __init__(self, args):

        self.args = args
        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)

        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.5e-3)

        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def td_error(self, state, action, next_state, reward, done):
        next_action = self.actor_target.forward(next_state)
        next_q = self.critic_target(next_state, next_action)

        done = 1 if done else 0
        if self.args.use_done_mask: next_q = next_q * (1 - done)  # Done mask
        target_q = reward + (self.gamma * next_q)

        current_q = self.critic(state, action)
        dt = (current_q - target_q).abs()
        return dt.item()

    def update_parameters(self, batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

        # Load everything to GPU if not already
        self.actor_target.to(self.args.device)
        self.critic_target.to(self.args.device)
        self.critic.to(self.args.device)
        state_batch = state_batch.to(self.args.device)
        next_state_batch = next_state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_batch = reward_batch.to(self.args.device)
        if self.args.use_done_mask: done_batch = done_batch.to(self.args.device)

        # Critic Update
        next_action_batch = self.actor_target.forward(next_state_batch)
        next_q = self.critic_target.forward(next_state_batch, next_action_batch)
        if self.args.use_done_mask: next_q = next_q * (1 - done_batch) #Done mask
        target_q = reward_batch + (self.gamma * next_q).detach()

        self.critic_optim.zero_grad()
        current_q = self.critic.forward(state_batch, action_batch)
        delta = (current_q - target_q).abs()
        dt = torch.mean(delta**2)
        dt.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        # Actor Update
        self.actor_optim.zero_grad()

        policy_grad_loss = -(self.critic.forward(state_batch, self.actor.forward(state_batch))).mean()
        policy_loss = policy_grad_loss

        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_grad_loss.data.cpu().numpy(), delta.data.cpu().numpy()



