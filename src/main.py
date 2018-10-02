import torch

# Config
device = torch.device('cuda')

world = gym.make('CartPole-v0').unwrapped
screen_width = 600

batch_size = 128
gamma = 0.999
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
target_decay = 10