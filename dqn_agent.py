import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, Double_DQN=False, Priority_Replay_Paras = False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.BUFFER_SIZE = BUFFER_SIZE
        # setting optional extra techniques
        self.Double_DQN = Double_DQN
        self.prio_e, self.prio_a, self.prio_b = Priority_Replay_Paras

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, Double_DQN, Priority_Replay_Paras)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, experience_indexes, priorities = self.memory.sample()
                self.learn(experiences, experience_indexes, priorities, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, experience_indexes, priorities, gamma):
        """Update value parameters using given batch of experience tuples.
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## compute and minimize the loss
        
        # calculate current Q_sa
        Q_s = self.qnetwork_local(states)
        Q_s_a = self.qnetwork_local(states).gather(1, actions)


        # Get max predicted Q values (for next states) from target model
        if self.Double_DQN:
            # double DQN uses the local network for selecting best action and evaluates it with target network
            best_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
            Q_s_next = self.qnetwork_target(next_states).gather(1, best_actions)
        else:
            Q_s_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
            
        targets = rewards + gamma * Q_s_next * (1 - dones)
        
        # calculate loss between the two
        losses = (Q_s_a - targets)**2
        
        # importance-sampling weights aka formula from Prioritized Experience Replay
        importance_weights = (((1/self.BUFFER_SIZE)*(1/priorities))**self.prio_b).unsqueeze(1)
        
        loss = (importance_weights*losses).mean()

        # calculate gradients and do a step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # calculate priorities and update them
        target_priorities = abs(Q_s_a - targets).detach().cpu().numpy() + self.prio_e
        self.memory.update_priority(experience_indexes, target_priorities)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def update_beta(self, interpolation):
        """Update priority beta for unbiased Q updates.

        Params
        ======
            interpolation (float): number between 0 and 1 specifying how much to interpolate to beta = 1
        """
        self.prio_b += (1 - self.prio_b)*interpolation


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, Double_DQN, Priority_Replay_Paras):
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
        self.priority = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.prio_e, self.prio_a, self.prio_b = Priority_Replay_Paras
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priority.append(self.prio_e)
        
    def update_priority(self, priority_indexes, priority_targets):
        for index,priority_index in enumerate(priority_indexes):
            self.priority[priority_index] = priority_targets[index][0]
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # using formulas for prioritised experience replay
        adjusted_priority = np.array(self.priority)**self.prio_a
        sampling_probability = adjusted_priority/sum(adjusted_priority)
        experience_indexes = np.random.choice(np.arange(len(self.priority)), size=self.batch_size, replace=False, p=sampling_probability)
        experiences = [self.memory[index] for index in experience_indexes]
        #priorities = [self.priority[index] for index in experience_indexes]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.array([self.priority[index] for index in experience_indexes])).float().to(device)
  
        return (states, actions, rewards, next_states, dones), experience_indexes, priorities

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)