from evaluation import *

from agent import QLearning
import matplotlib.pyplot as plt
from maze import *
from tqdm import tqdm
import gymnasium as gym

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

grids_per_dim = 4 # 10 points per state dimension
lr = 0.02
epsilon = 0.05
n_episodes = 5000
discount = 0.99

eval_step = []
eval_reward = []
eval_error = []

# https://gymnasium.farama.org/environments/classic_control/acrobot/
env = gym.make('Acrobot-v1')
name = env.unwrapped.spec.id
env.action_space.seed(123) # set seed for reproducibility

q_learner = QLearning(env, lr=lr, discount=discount, epsilon=epsilon, grids_per_dim=grids_per_dim)


for i in tqdm(range(n_episodes)):  
	state, _ = env.reset()
	done = False
	while not done:
		action = q_learner.get_action_egreedy(state)
		next_state, reward, terminated, truncated, info = env.step(action)

        # update the agent
		q_learner.update(state, action, reward, next_state)

        # update if the environment is done and the current obs
		done = terminated or truncated
		state = next_state

	# evaluate the agent using the Q-table every 50 steps
	if (i+1) % 50 == 0:
		avg_step, avg_reward = evaluation(env, q_learner)
		# print(f'Episode {i+1}, Average steps to goal: {avg_step}, Average reward: {avg_reward}')
		eval_step.append(avg_step)
		eval_reward.append(avg_reward)

f1, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(range(0, n_episodes, 50), eval_step)
axes[0].set_title('Average steps to goal in an episode')

axes[1].plot(range(0, n_episodes, 50), eval_reward)
axes[1].set_title('Average reward in an episode')

plt.tight_layout()
plt.savefig(f'plots/Q_{name}.png')