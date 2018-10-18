import environment
import model
import torch
from itertools import count
from settings import DEVICE, SCREEN_WIDTH, TARGET_UPDATE, EPOCHS

# Environment Setup
world = environment.init()
world.reset()

# Log how long our agent lasts for each iteration
durations = []

# Training
agent = model.Agent(DEVICE)

for i in range(EPOCHS):
	# Initialize environment 
	world.reset()

	# Get current state
	last_screen = environment.get_screen(world, SCREEN_WIDTH, DEVICE)
	current_screen = environment.get_screen(world, SCREEN_WIDTH, DEVICE)
	state = current_screen - last_screen

	for t in count():
		# Select and perform an action
		action = agent.select_action(state)
		_, reward, done, _ = world.step(action.item())
		reward = torch.tensor([reward], device=DEVICE)

		# Observe new state
		last_screen = current_screen
		current_screen = environment.get_screen(world, SCREEN_WIDTH, DEVICE)

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
	if i % TARGET_UPDATE == 0:
		agent.target_net.load_state_dict(agent.policy_net.state_dict())

world.render()
world.close()

print(durations)
