import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, layers):
        super(Actor, self).__init__()

        # self.l1 = nn.Linear(state_dim, 400)
        # self.l2 = nn.Linear(400, 300)
        # self.l3 = nn.Linear(300, action_dim)

        self.layers = [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.layers.append(nn.Linear(layers[-1], action_dim))

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.layers[0](state))
        for l in self.layers[1:-1]:
            a = F.relu(l(a))
        return self.max_action * torch.tanh(self.layers[-1](a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layers):
        super(Critic, self).__init__()

        self.layers = [nn.Linear(state_dim + action_dim, layers[0])]
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.layers.append(nn.Linear(layers[-1], 1))

    def forward(self, state, action):
        q = F.relu(self.layers[0](torch.cat([state, action], 1)))
        for l in self.layers[1:-1]:
            q = F.relu(l(q))
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.layers[-1](q)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0., layers=None):
        if layers is None:
            layers = [400, 300]
        self.actor = Actor(state_dim, action_dim, max_action, layers).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim, layers).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.discount = discount
        self.tau = tau
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic, filename + "_critic")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor, filename + "_actor")
        # torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
