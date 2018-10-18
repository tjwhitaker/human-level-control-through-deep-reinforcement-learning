import gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Resize frames we grab from gym and convert to tensor
resizer = T.Compose([
	T.ToPILImage(),
	T.Resize(40, interpolation=Image.CUBIC),
	T.ToTensor()
])

# Start cartpole application through gym
def init():
	return gym.make('CartPole-v0').unwrapped

# Get cart location with respect to center
def get_cart_location(world, screen_width):
	world_width = world.x_threshold * 2
	scale = screen_width / world_width
	return int(world.state[0] * scale + screen_width / 2.0)

# Get screen tensor from gym application
def get_screen(world, screen_width, device):
	screen = world.render(mode='rgb_array').transpose((2, 0, 1))
	screen = screen[:, 160:320]
	view_width = 320
	cart_location = get_cart_location(world, screen_width)

	if cart_location < view_width // 2:
		slice_range = slice(view_width)
	elif cart_location > (screen_width - view_width // 2):
		slice_range = slice(-view_width, None)
	else:
		slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

	screen = screen[:, :, slice_range]
	screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
	screen = torch.from_numpy(screen)
	return resizer(screen).unsqueeze(0).to(device)