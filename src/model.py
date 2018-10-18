import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from settings import EPSILON_START, EPSILON_END, EPSILON_DECAY, GAMMA, BATCH_SIZE

# Memory representation of states
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Deep Q Network
class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		self.head = nn.Linear(448, 2)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

# Memory representation for our agent
class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = deque(maxlen=capacity)

	def push(self, *args):
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

# Agent that performs, remembers and learns actions
class Agent():
	def __init__(self, device):
		self.device = device
		self.policy_net = DQN().to(self.device)
		self.target_net = DQN().to(self.device)
		self.optimizer = optim.RMSprop(self.policy_net.parameters())
		self.memory = ReplayMemory(10000)
		self.steps_done = 0
		
	def remember(self, *args):
		self.memory.push(*args)

	def select_action(self, state):
		# Select an action according to an epsilon greedy approach
		sample = random.random()
		epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.steps_done / EPSILON_DECAY) 
		self.steps_done += 1
		if sample < epsilon_threshold:
			return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
		else:
			with torch.no_grad():
				return self.policy_net(state).max(1)[1].view(1, 1)

	def optimize_model(self):
		if len(self.memory) < BATCH_SIZE:
			return

		# Sample from our memory
		transitions = self.memory.sample(BATCH_SIZE)
		batch = Transition(*zip(*transitions))

		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
		non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

		# Concatenate our tensors
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		state_action_values = self.policy_net(state_batch).gather(1, action_batch)

		next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
		next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

		expected_state_action_values = (next_state_values * GAMMA) + reward_batch

		# Compute loss between our state action and expectations
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		self.optimizer.zero_grad()
		loss.backward()

		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1)

		self.optimizer.step()
		