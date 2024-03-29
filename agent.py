import numpy as np
import random
import copy
from collections import namedtuple, deque
from models import Actor, Critic
from noise import NoiseReducer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiActorCriticAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, agent_parameters):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            agent_parameters (AgentParameters) : parameters used to train the agent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = agent_parameters.gamma
        self.tau, self.tau_end, self.tau_decay = agent_parameters.tau_param
        self.batch_size = agent_parameters.batch_size
        self.update_every, self.update_count = agent_parameters.model_update
        self.name = f'agent'

        lr_actor, lr_critic = agent_parameters.lr_actor_critic
        # Actor Networks (local one and target one)
        fc1_units, fc2_units = agent_parameters.model_param
        self.actor_local = Actor(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units).to(device)
        self.actor_target = Actor(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (local one and target one)
        self.critic_local = Critic(state_size, action_size, seed, fcs1_units=fc1_units, fc2_units=fc2_units).to(device)
        self.critic_target = Critic(state_size, action_size, seed, fcs1_units=fc1_units, fc2_units=fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

        # Initialize the target model weights with the local ones (same values)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # Noise process
        factor_reduction, min_factor, rate_reduction = agent_parameters.noise_reducer_param
        self.noise = agent_parameters.noise
        self.noise_reducer = NoiseReducer(factor_reduction, min_factor, rate_reduction)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, agent_parameters.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):

        # Save experience in replay memory
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                for _ in range(self.update_count):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy.

        Params
        ======
            states (array_like): current state
            add_noise: indicates if noise should be added
        """
        states = torch.from_numpy(states).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:
            actions += self.noise_reducer.reduce_noise(self.noise.sample())

        return np.clip(actions, -1, 1)

    def end_episode(self):
        """ Method applied at the end of each episode """
        self.tau = max(self.tau_end, self.tau*self.tau_decay)
        self.noise_reducer.update_factor()

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # as suggested in the "Benchmak implementation" section of the course"
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
