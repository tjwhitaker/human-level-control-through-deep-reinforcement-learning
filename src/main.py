import environment
import model
import torch

from itertools import count

# Config
screen_width = 600
target_update = 10
iterations = 500
durations = []

device = torch.device('cuda')

# Environment Setup
world = environment.init()
world.reset()

# Training
agent = model.Agent(device)

for i in range(iterations):
	# Initialize environment 
	world.reset()

	# Get current state
	last_screen = environment.get_screen(world, screen_width, device)
	current_screen = environment.get_screen(world, screen_width, device)
	state = current_screen - last_screen

	for t in count():
		# Select and perform an action
		action = agent.select_action(state)
		_, reward, done, _ = world.step(action.item())
		reward = torch.tensor([reward], device=device)

		# Observe new state
		last_screen = current_screen
		current_screen = environment.get_screen(world, screen_width, device)

		if not done:
			next_state = current_screen - last_screen
		else:
			next_state = None

		# Store the transition in memory
		agent.remember(state, action, next_state, reward)

		# Move to the next state
		state = next_state

		# Optimize the target network
		agent.optimize_model()

		if done:
			durations.append(t + 1)
			break

	# Update the target network
	if i % target_update == 0:
		agent.target_net.load_state_dict(agent.policy_net.state_dict())

world.render()
world.close()

print(durations)
