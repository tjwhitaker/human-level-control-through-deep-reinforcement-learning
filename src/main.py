import torch

# Config
screen_width = 600
batch_size = 128
gamma = 0.999
epsilon_start = 0.9
epsilon_end = 0.05
epsilon_decay = 200
target_decay = 10

device = torch.device('cuda')

world = environment.init()