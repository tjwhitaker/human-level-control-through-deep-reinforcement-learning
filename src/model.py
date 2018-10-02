Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
	def __init__(self, capacity):

	def push(self, *args):

	def sample(self, batch_size):

	def __len__(self):

class DQN(nn.module):
	def __init__(self):

	def forward(self, x):