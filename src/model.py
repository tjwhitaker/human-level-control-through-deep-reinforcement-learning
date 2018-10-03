import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple

# Config
batch_size = 128
gamma = 0.999
epsilon_start = 0.9
epsilon_end = 0.05
epsilon_decay = 200

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
		sample = random.random()
		epsilon_threshold = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * self.steps_done / epsilon_decay) 
		self.steps_done += 1
		if sample < epsilon_threshold:
			return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
		else:
			with torch.no_grad():
				return self.policy_net(state).max(1)[1].view(1, 1)

	def optimize_model(self):
		if len(self.memory) < batch_size:
			return

		transitions = self.memory.sample(batch_size)
		batch = Transition(*zip(*transitions))

		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
		non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		state_action_values = self.policy_net(state_batch).gather(1, action_batch)

		next_state_values = torch.zeros(batch_size, device=self.device)
		next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

		expected_state_action_values = (next_state_values * gamma) + reward_batch

		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		self.optimizer.zero_grad()
		loss.backward()

		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1)

		self.optimizer.step()
