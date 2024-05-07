import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import numpy as np
from agent import QLearning
import random


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def evaluation(env, q_learner, step_bound = 400, num_itr = 10):
	"""
	Semi-greedy evaluation for discrete state and discrete action spaces and an episodic environment.

	Input:
		env : an environment object. 
		Q : A numpy array. Q values for all state and action pairs. 
			Q.shape = (the number of states, the number of actions)
		step_bound : the maximum number of steps for each iteration
		num_itr : the number of iterations

	Output:
		Total number of steps taken to finish an episode (averaged over num_itr trials)
		Cumulative reward in an episode (averaged over num_itr trials)

	"""
	total_step = 0 
	total_reward = 0 
	itr = 0 
	while(itr<num_itr):
		steps = 0
		np.random.seed()
		state, _ = env.reset()
		rewards = 0.0
		done = False

		while((not done) and (steps < step_bound)):
			action = q_learner.get_action_egreedy(state)
			next_state, reward, terminated, truncated, info = env.step(action)
			state = next_state

			rewards += reward
			steps +=1
			done = terminated or truncated

		total_reward += rewards
		total_step += steps
		itr += 1

	return total_step/float(num_itr), total_reward/float(num_itr)


########################################################################
# Environment setup
# Uncomment as needed

# Acrobat-V1
env = gym.make('Acrobot-v1')
grids_per_dim = 2 # 10 points per state dimension
step_bound = 400

init_lr = 0.1
init_epsilon = 0.05
lr_decay = 0.95
epsilon_decay = 0.95
decay_step = 100
n_episodes = 2000
discount = 0.99
seed = 3

# # MountainCar-V0
# env = gym.make('MountainCar-v0', max_episode_steps=1000)
# grids_per_dim = 2 # 10 points per state dimension
# step_bound = 1000

# init_lr = 0.1
# init_epsilon = 0.05
# lr_decay = 0.9
# epsilon_decay = 0.9
# decay_step = 100
# n_episodes = 2000
# discount = 0.99
# seed = 3
########################################################################

eval_step = []
eval_reward = []
eval_error = []

random.seed(seed)
np.random.seed(seed)

name = env.unwrapped.spec.id

q_learner = QLearning(env, lr=init_lr, discount=discount, epsilon=init_epsilon, grids_per_dim=grids_per_dim)

for i in tqdm(range(n_episodes)):  
	# evaluate the agent using the Q-table every 50 steps
	avg_step, avg_reward = evaluation(env, q_learner, step_bound=step_bound)
	eval_step.append(avg_step)
	eval_reward.append(avg_reward)

	state, _ = env.reset(seed=seed)
	done = False
	while not done:
		
		action = q_learner.get_action_egreedy(state)
		next_state, reward, terminated, truncated, info = env.step(action)

        # update the agent
		q_learner.update(state, action, reward, next_state)

        # update if the environment is done and the current obs
		done = terminated or truncated
		state = next_state

	if i % decay_step == 0:
		print(f'Episode {i}, Average steps to goal: {avg_step}, Average reward: {avg_reward}. LR: {q_learner.lr:.4f}, Epsilon: {q_learner.epsilon:.4f}')

		# new_lr = max(q_learner.lr * epsilon_decay, 1e-4)
		# new_eps = max(q_learner.epsilon * lr_decay, 1e-4)
		new_lr = q_learner.lr * lr_decay
		new_eps = q_learner.epsilon * epsilon_decay
		q_learner.update_lr(new_lr)
		q_learner.update_epsilon(new_eps)

# last epoch
avg_step, avg_reward = evaluation(env, q_learner, step_bound=step_bound)
eval_step.append(avg_step)
eval_reward.append(avg_reward)
print(f'Episode {i}, Average steps to goal: {avg_step}, Average reward: {avg_reward}')



f1, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(range(0, n_episodes+1), eval_step)
axes[0].set_title('Average steps to goal in an episode')

axes[1].plot(range(0, n_episodes+1), eval_reward)
axes[1].set_title('Average reward in an episode')

plt.tight_layout()
plt.savefig(f'plots/Q_{name}_{lr_decay}_{epsilon_decay}_{decay_step}.png')